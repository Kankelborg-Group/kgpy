import dataclasses
import typing as typ
import pathlib
import win32com.client.gencache
import numpy as np
import astropy.units as u
from kgpy import optics
from .. import ZOSAPI
from . import configuration, wavelengths, util, fields, surface

__all__ = ['System', 'calc_zemax_system']


def load_zemax_app() -> ZOSAPI.IZOSAPI_Application:
    zemax_connection = win32com.client.gencache.EnsureDispatch('ZOSAPI.ZOSAPI_Connection')
    if zemax_connection is None:
        raise ValueError('Unable to initialize COM connection to ZOSAPI')

    zemax_app = zemax_connection.CreateNewApplication()
    if zemax_app is None:
        raise ValueError('Unable to acquire ZOSAPI application')
    if not zemax_app.IsValidLicenseForAPI:
        raise ValueError('Invalid licence (Possibly too many instances of OpticStudio are open).')

    return zemax_app


@dataclasses.dataclass
class InstanceVarBase:

    _entrance_pupil_radius_op: configuration.Operand = dataclasses.field(
        default_factory=lambda: configuration.Operand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.APER,
        ),
        init=False,
        repr=False,
    )

    _stop_surface_index_op: configuration.Operand = dataclasses.field(
        default_factory=lambda: configuration.Operand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.STPS
        ),
        init=False,
        repr=False,
    )

    _zemax_app: ZOSAPI.IZOSAPI_Application = dataclasses.field(default_factory=load_zemax_app, init=False, repr=False, )

    _lens_units: u.Unit = dataclasses.field(default_factory=lambda: u.mm, repr=False)

    _mce_: configuration.Editor = dataclasses.field(
        default_factory=lambda: configuration.Editor(), init=False, repr=False,
    )


SurfacesT = typ.TypeVar('SurfacesT', bound='typ.Iterable[surface.Surface]')


@dataclasses.dataclass
class System(optics.System[SurfacesT], InstanceVarBase):

    def __post_init__(self):
        self._mce = self._mce

    def save(self, filename: pathlib.Path):
        self._zemax_system.SaveAs(str(filename))

    @property
    def surfaces(self) -> SurfacesT:
        return self._surfaces

    @surfaces.setter
    def surfaces(self, value: SurfacesT):
        num_surfaces = len(list(value)) + 1
        while self._lde.NumberOfSurfaces != num_surfaces:
            if self._lde.NumberOfSurfaces < num_surfaces:
                self._lde.AddSurface()
            else:
                self._lde.RemoveSurfaceAt(self._lde.NumberOfSurfaces)

        self._surfaces = value
        for v in value:
            v._composite = self

    def _entrance_pupil_radius_setter(self, value: float):
        self._zemax_system.SystemData.Aperture.ApertureType = ZOSAPI.SystemData.ZemaxApertureType.EntrancePuilDiameter
        self._zemax_system.SystemData.Aperture.ApertureValue = value

    @property
    def entrance_pupil_radius(self) -> u.Quantity:
        return self._entrance_pupil_radius

    @entrance_pupil_radius.setter
    def entrance_pupil_radius(self, value: u.Quantity):
        self._entrance_pupil_radius = value
        self._set(value, self._entrance_pupil_radius_setter, self._entrance_pupil_radius_op, self._lens_units)

    def _stop_surface_index_setter(self, value: int):
        surf = list(self.surfaces)[value]
        surf._lde_row.IsStop = True

    @property
    def stop_surface_index(self) -> int:
        return self._stop_surface_index

    @stop_surface_index.setter
    def stop_surface_index(self, value: int):
        self._stop_surface_index = value
        self._set(value, self._stop_surface_index_setter, self._stop_surface_index_op)

    @property
    def _zemax_system(self) -> ZOSAPI.IOpticalSystem:
        return self._zemax_app.PrimarySystem

    @property
    def _lde(self) -> ZOSAPI.Editors.LDE.ILensDataEditor:
        return self._zemax_system.LDE

    @property
    def _mce(self) -> configuration.Editor:
        return self._mce_

    @_mce.setter
    def _mce(self, value: configuration.Editor):
        self._mce_ = value
        value._composite = self

    @property
    def _lens_units(self) -> u.Unit:
        return self.__lens_units

    @_lens_units.setter
    def _lens_units(self, value: u.Unit):
        self.__lens_units = value

        if value == u.mm:
            self._zemax_system.SystemData.Units.LensUnits = ZOSAPI.SystemData.ZemaxSystemUnits.Millimeters
        elif value == u.cm:
            self._zemax_system.SystemData.Units.LensUnits = ZOSAPI.SystemData.ZemaxSystemUnits.Centimeters
        elif value == u.m:
            self._zemax_system.SystemData.Units.LensUnits = ZOSAPI.SystemData.ZemaxSystemUnits.Meters
        elif value == u.imperial.inch:
            self._zemax_system.SystemData.Units.LensUnits = ZOSAPI.SystemData.ZemaxSystemUnits.Inches
        else:
            raise ValueError('Unsupported unit')

    def _set(
            self,
            value: typ.Any,
            setter: typ.Callable[[typ.Any], None],
            operand: configuration.Operand,
            unit: u.Unit = None,
    ) -> typ.NoReturn:
        print('here')
        if unit is not None:
            value = value.to(unit).value

        if np.isscalar(value):
            if operand._composite is not None:
                self._mce.pop(operand.mce_index)
            setter(value)

        else:
            if operand._composite is None:
                self._mce.append(operand)
            operand.data = value


def calc_zemax_system(system: 'optics.System') -> typ.Tuple[ZOSAPI.IOpticalSystem, u.Unit]:
    zemax_system = open_zemax_system()

    # zemax_system.SystemData.Units = ZOSAPI.SystemData.ZemaxSystemUnits.Millimeters
    zemax_lens_units = u.mm

    configuration_size = system.config_broadcast.size
    configuration_shape = system.config_broadcast.shape

    while zemax_system.MCE.NumberOfConfigurations < configuration_size:
        zemax_system.MCE.AddConfiguration(False)

    set_entrance_pupil_radius(zemax_system, system.entrance_pupil_radius, configuration_shape, zemax_lens_units)
    fields.add_to_zemax_system(zemax_system, system.field_grid, configuration_shape)
    wavelengths.add_to_zemax_system(zemax_system, system.wavelengths, configuration_shape)
    surface.add_surfaces_to_zemax_system(zemax_system, system.surfaces, configuration_shape, zemax_lens_units)

    set_stop_surface(zemax_system, system.stop_surface_index, configuration_shape)

    return zemax_system, zemax_lens_units


def open_zemax_system() -> ZOSAPI.IOpticalSystem:
    # Create COM connection to Zemax
    zemax_connection = win32com.client.gencache.EnsureDispatch(
        'ZOSAPI.ZOSAPI_Connection')  # type: ZOSAPI.ZOSAPI_Connection
    if zemax_connection is None:
        raise ValueError('Unable to initialize COM connection to ZOSAPI')
    # Open Zemax system
    zemax_app = zemax_connection.CreateNewApplication()
    if zemax_app is None:
        raise ValueError('Unable to acquire ZOSAPI application')
    # Check if license is valid
    if not zemax_app.IsValidLicenseForAPI:
        raise ValueError('License is not valid for ZOSAPI use (Possibly too many instances of OpticStudio are open).')
    zemax_system = zemax_app.PrimarySystem
    if zemax_system is None:
        raise ValueError('Unable to acquire Primary system')
    return zemax_system


def set_entrance_pupil_radius(
        zemax_system: ZOSAPI.IOpticalSystem,
        entrance_pupil_radius: u.Quantity,
        configuration_shape: typ.Tuple[int],
        zemax_units: u.Unit
):
    zemax_system.SystemData.Aperture.ApertureType = ZOSAPI.SystemData.ZemaxApertureType.EntrancePuilDiameter

    op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.APER

    entrance_pupil_diameter = 2 * entrance_pupil_radius
    util.set_float(zemax_system, entrance_pupil_diameter, configuration_shape, op_type, zemax_unit=zemax_units)


def set_stop_surface(
        zemax_system: ZOSAPI.IOpticalSystem,
        stop_surface_index: int,
        configuration_shape: typ.Tuple[int],
):
    op_stop_surface_index = ZOSAPI.Editors.MCE.MultiConfigOperandType.STPS
    util.set_int(zemax_system, stop_surface_index, configuration_shape, op_stop_surface_index)

import dataclasses
import typing as typ
import win32com.client.gencache
import numpy as np
import astropy.units as u

from kgpy import optics

from .. import ZOSAPI
from . import configuration, wavelengths, util, fields, surface

__all__ = ['SystemT', 'System', 'calc_zemax_system']

SystemT = typ.TypeVar('SystemT', bound='System')


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

    zemax_app: ZOSAPI.IZOSAPI_Application = dataclasses.field(
        default_factory=load_zemax_app,
        init=False,
        repr=False,
    )

    mce: configuration.Editor = dataclasses.field(
        default_factory=lambda: configuration.Editor(),
        init=False,
        repr=False,
    )


@dataclasses.dataclass
class Base(optics.System, InstanceVarBase):

    lens_units: u.Unit = u.mm


class System(Base):

    @property
    def zemax_system(self) -> ZOSAPI.IOpticalSystem:
        return self.zemax_app.PrimarySystem

    @property
    def mce(self) -> configuration.Editor:
        return self._config_operands

    @mce.setter
    def mce(self, value: configuration.Editor):
        self._config_operands = value
        value.system = self

    @property
    def lens_units(self) -> u.Unit:
        return self.lens_units

    @lens_units.setter
    def lens_units(self, value: u.Unit):
        self.lens_units = value

        if value == u.mm:
            self.zemax_system.SystemData.Units = ZOSAPI.SystemData.ZemaxSystemUnits.Millimeters
        elif value == u.cm:
            self.zemax_system.SystemData.Units = ZOSAPI.SystemData.ZemaxSystemUnits.Centimeters
        elif value == u.m:
            self.zemax_system.SystemData.Units = ZOSAPI.SystemData.ZemaxSystemUnits.Meters
        elif value == u.imperial.inch:
            self.zemax_system.SystemData.Units = ZOSAPI.SystemData.ZemaxSystemUnits.Inches
        else:
            raise ValueError('Unsupported unit')

    def set(
            self,
            value: typ.Any,
            setter: typ.Callable[[typ.Any], None],
            operand: configuration.Operand,
            unit: u.Unit = None,
    ):
        if unit is not None:
            value = value.to(unit).value

        if np.isscalar(value):
            if operand.mce is not None:
                self.mce.pop(operand.mce_index)
            setter(value)

        else:
            if operand.mce is None:
                self.mce.append(operand)
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
    fields.add_to_zemax_system(zemax_system, system.fields, configuration_shape)
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

import dataclasses
import typing as typ
import win32com.client
from astropy import units as u
from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from .. import configuration
from . import Standard

__all__ = ['DiffractionGrating']


@dataclasses.dataclass
class InstanceVarBase:
    _groove_frequency_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=1,
        ),
        init=None,
        repr=None,
    )
    _diffraction_order_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=2
        ),
        init=None,
        repr=None,

    )
    _groove_frequency_unit: typ.ClassVar[u.Unit] = 1 / u.um
    _diffraction_order_unit: typ.ClassVar[u.Unit] = u.dimensionless_unscaled


@dataclasses.dataclass
class DiffractionGrating(system.surface.DiffractionGrating, InstanceVarBase, Standard, ):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.groove_frequency = self.groove_frequency
        self.diffraction_order = self.diffraction_order

    @property
    def _lde_row_type(self) -> ZOSAPI.Editors.LDE.SurfaceType:
        return ZOSAPI.Editors.LDE.SurfaceType.DiffractionGrating

    @property
    def _lde_row_data(self) -> ZOSAPI.Editors.LDE.ISurfaceDiffractionGrating:
        return win32com.client.CastTo(self._lde_row.SurfaceData, ZOSAPI.Editors.LDE.ISurfaceDiffractionGrating.__name__)

    @property
    def _lde_row(self) -> ZOSAPI.Editors.LDE.ILDERow[ZOSAPI.Editors.LDE.ISurfaceDiffractionGrating]:
        return super()._lde_row

    def _groove_frequency_setter(self, value: float):
        self._lde_row_data.LinesPerMicroMeter = value

    @property
    def groove_frequency(self) -> u.Quantity:
        return self._groove_frequency

    @groove_frequency.setter
    def groove_frequency(self, value: u.Quantity):
        self._groove_frequency = value
        self._set(value, self._groove_frequency_setter, self._groove_frequency_op, self._groove_frequency_unit)

    def _diffraction_order_setter(self, value: float):
        self._lde_row_data.DiffractionOrder = value

    @property
    def diffraction_order(self) -> u.Quantity:
        return self._diffraction_order

    @diffraction_order.setter
    def diffraction_order(self, value: u.Quantity):
        self._diffraction_order = value
        self._set(value, self._diffraction_order_setter, self._diffraction_order_op, self._diffraction_order_unit)

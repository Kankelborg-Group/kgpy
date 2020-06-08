import dataclasses
import typing as typ
from astropy import units as u
import win32com.client
from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from .. import configuration
from . import Standard

__all__ = ['Toroidal']


@dataclasses.dataclass
class InstanceVarBase:
    _radius_of_rotation_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=1,
        ),
        init=None,
        repr=None,
    )


@dataclasses.dataclass
class Toroidal(system.surface.Toroidal, InstanceVarBase, Standard, ):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.radius_of_rotation = self.radius_of_rotation

    @property
    def _lde_row_type(self) -> ZOSAPI.Editors.LDE.SurfaceType:
        return ZOSAPI.Editors.LDE.SurfaceType.Toroidal

    @property
    def _lde_row_data(self) -> ZOSAPI.Editors.LDE.ISurfaceToroidal:
        return win32com.client.CastTo(self._lde_row.SurfaceData, ZOSAPI.Editors.LDE.ISurfaceToroidal.__name__)

    @property
    def _lde_row(self) -> ZOSAPI.Editors.LDE.ILDERow[ZOSAPI.Editors.LDE.ISurfaceToroidal]:
        return super()._lde_row

    def _radius_of_rotation_setter(self, value: float):
        self._lde_row_data.RadiusOfRotation = value

    @property
    def radius_of_rotation(self) -> u.Quantity:
        return self._radius_of_rotation

    @radius_of_rotation.setter
    def radius_of_rotation(self, value: u.Quantity):
        self._radius_of_rotation = value
        self._set_with_lens_units(value, self._radius_of_rotation_setter, self._radius_of_rotation_op)

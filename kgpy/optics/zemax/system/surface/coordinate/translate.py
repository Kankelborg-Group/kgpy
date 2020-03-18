import dataclasses
import typing as typ
from astropy import units as u
from kgpy.optics.system import coordinate
from .... import ZOSAPI
from ... import configuration
from .parent import ParentT, ParentBase
from .decenter import Decenter

__all__ = ['Translate']


@dataclasses.dataclass
class OperandBase:

    _z_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC
        ),
        init=None,
        repr=None,
    )


class Translate(typ.Generic[ParentT], ParentBase[ParentT], coordinate.Translate, OperandBase):
    
    def _update(self) -> typ.NoReturn:
        self.decenter = self.decenter
        self.z = self.z
    
    def _z_setter(self, value: float):
        self.transform.surface.lde_row.Thickness = value

    @property
    def decenter(self) -> Decenter[ParentT]:
        return self._decenter

    @decenter.setter
    def decenter(self, value: Decenter[ParentT]):
        value.parent = self.parent
        self._decenter = value

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self._z_op.surface_index = self.transform.surface.lde_index
            self.parent.parent.lde.system.set(value, self._z_setter, self._z_op, self.transform.surface.lens_units)
        except AttributeError:
            pass
        
    @property
    def parent(self) -> ParentT:
        return self._parent

    @parent.setter
    def parent(self, value: ParentT):
        self._parent = value
        self._update()

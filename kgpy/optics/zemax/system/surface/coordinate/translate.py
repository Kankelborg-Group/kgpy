import dataclasses
import typing as typ
from astropy import units as u
from kgpy.optics.system.surface import coordinate
from .... import ZOSAPI
from ... import Child, configuration, surface
from .decenter import Decenter

__all__ = ['Translate']

SurfaceChildT = typ.TypeVar('SurfaceChildT', bound=Child[surface.Surface])


@dataclasses.dataclass
class OperandBase:

    _z_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC
        ),
        init=None,
        repr=None,
    )


class Translate(Decenter[SurfaceChildT], coordinate.Translate, OperandBase, typ.Generic[SurfaceChildT], ):
    
    def _update(self) -> typ.NoReturn:
        super()._update()
        self.z = self.z

    def _x_setter(self, value: float):
        raise NotImplementedError

    def _y_setter(self, value: float):
        raise NotImplementedError
    
    def _z_setter(self, value: float):
        self.parent.parent.lde_row.Thickness = value

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self.parent.parent.set(value, self._z_setter, self._z_op, self.parent.parent.lens_units)
        except AttributeError:
            pass

import dataclasses
from astropy import units as u

from kgpy.optics.system import coordinate
from .... import ZOSAPI
from ... import configuration
from . import transform

__all__ = ['Translate']


@dataclasses.dataclass
class InstanceVarBase:

    _z_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC
        ),
        init=None,
        repr=None,
    )


@dataclasses.dataclass
class Base(coordinate.Translate, InstanceVarBase):
    
    transform: 'transform.Transform' = None


class Translate(Base):
    
    def _update(self) -> None:
        self.z = self.z
    
    def _z_setter(self, value: float):
        self.transform.surface.lde_row.Thickness = value

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self._z_op.surface_index = self.transform.surface.lde_index
            self.transform.surface.lde.system.set(value, self._z_setter, self._z_op, self.transform.surface.lens_units)
        except AttributeError:
            pass
        
    @property
    def transform(self) -> 'transform.Transform':
        return self._transform
    
    @transform.setter
    def transform(self, value: 'transform.Transform'):
        self._transform = value
        self._update()

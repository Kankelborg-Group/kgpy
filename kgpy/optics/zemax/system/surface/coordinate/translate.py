import dataclasses
from astropy import units as u

from kgpy.optics.system import coordinate
from .... import ZOSAPI
from ... import configuration
from .transform import Transform

__all__ = ['Translate']


@dataclasses.dataclass
class InstanceVarBase:

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=1,
        ),
        init=None,
        repr=None,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC,
            param_2=2,
        ),
        init=None,
        repr=None,
    )
    _z_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC
        ),
        init=None,
        repr=None,
    )


@dataclasses.dataclass
class Base(coordinate.Translate, InstanceVarBase):
    
    transform: Transform = None


class Translate(Base):
    
    def _update(self) -> None:
        self.z = self.z

    def _x_setter(self, value: float):
        self.transform.surface.lde_row.TiltDecenterData.BeforeSurfaceDecenterX = value
    
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
    def transform(self) -> Transform:
        return self._transform
    
    @transform.setter
    def transform(self, value: Transform):
        self._transform = value
        self._update()

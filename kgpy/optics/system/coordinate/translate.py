import dataclasses
import numpy as np
from astropy import units as u

from kgpy.optics.system import mixin

from . import Decenter

__all__ = ['Translate', 'InverseTranslate']


@dataclasses.dataclass
class Base(mixin.ConfigBroadcast):
    
    decenter: Decenter = dataclasses.field(init=False, repr=False, default_factory=lambda: Decenter())
    
    x: u.Quantity = 0 * u.mm
    y: u.Quantity = 0 * u.mm
    z: u.Quantity = 0 * u.mm
    
    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.x,
            self.y,
            self.z,
        )

    def __invert__(self):
        return type(self)(
            -self.x,
            -self.y,
            -self.z,
        )


class Translate(Base):

    @property
    def x(self) -> u.Quantity:
        return self.decenter.x

    @x.setter
    def x(self, value: u.Quantity):
        self.decenter.x = value

    @property
    def y(self) -> u.Quantity:
        return self.decenter.y

    @y.setter
    def y(self, value: u.Quantity):
        self.decenter.y = value
        

@dataclasses.dataclass
class InverseTranslate:
    
    _translate: Translate
    
    @property
    def x(self) -> u.Quantity:
        return -self._translate.x
    
    @property
    def y(self) -> u.Quantity:
        return -self._translate.y
    
    @property
    def z(self) -> u.Quantity:
        return -self._translate.z

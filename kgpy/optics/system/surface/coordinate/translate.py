import dataclasses
import numpy as np
from astropy import units as u
from ... import mixin
from . import Decenter, InverseDecenter

__all__ = ['Translate', 'InverseTranslate']


@dataclasses.dataclass
class Base(mixin.ConfigBroadcast):
    
    decenter: Decenter = dataclasses.field(init=False, repr=False, default_factory=lambda: Decenter())
    
    x: u.Quantity = 0 * u.mm
    y: u.Quantity = 0 * u.mm
    z: u.Quantity = 0 * u.mm


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

    def __add__(self, other: 'Translate'):
        return type(self)(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def copy(self):
        return type(self)(
            self.x.copy(),
            self.y.copy(),
            self.z.copy()
        )


@dataclasses.dataclass
class InverseTranslate:
    
    _translate: Translate

    @property
    def config_broadcast(self):
        return self._translate.config_broadcast

    @property
    def decenter(self):
        return InverseDecenter(self._translate.decenter)
    
    @property
    def x(self) -> u.Quantity:
        return -self._translate.x
    
    @property
    def y(self) -> u.Quantity:
        return -self._translate.y
    
    @property
    def z(self) -> u.Quantity:
        return -self._translate.z

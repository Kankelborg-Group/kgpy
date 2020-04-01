import dataclasses
import numpy as np
from astropy import units as u
from . import Decenter

__all__ = ['Translate', 'InverseTranslate']


@dataclasses.dataclass
class Base:

    decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter(), init=None, repr=None)


class Translate(Decenter, Base):

    z: u.Quantity = 0 * u.mm

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
    def x(self) -> u.Quantity:
        return -self._translate.x
    
    @property
    def y(self) -> u.Quantity:
        return -self._translate.y
    
    @property
    def z(self) -> u.Quantity:
        return -self._translate.z

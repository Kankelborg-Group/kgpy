import dataclasses
import numpy as np
from astropy import units as u

from kgpy.optics.system import mixin

from . import Decenter


@dataclasses.dataclass
class Translate(mixin.ConfigBroadcast):

    _decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter())
    z: u.Quantity = 0 * u.mm

    @classmethod
    def from_coords(cls, x: u.Quantity, y: u.Quantity, z: u.Quantity) -> 'Translate':
        self = cls()
        self.x = x
        self.y = y
        self.z = z
        return self

    @property
    def x(self) -> u.Quantity:
        return self._decenter.x

    @x.setter
    def x(self, value: u.Quantity):
        self._decenter.x = value

    @property
    def y(self) -> u.Quantity:
        return self._decenter.y

    @y.setter
    def y(self, value: u.Quantity):
        self._decenter.y = value

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self._decenter.config_broadcast,
            self.z,
        )

    def __invert__(self) -> 'Translate':
        return type(self)(
            self._decenter.__invert__(),
            -self.z,
        )

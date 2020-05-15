import dataclasses
import numpy as np
from astropy import units as u
from . import Decenter

__all__ = ['Translate']


@dataclasses.dataclass
class Translate(Decenter):

    z: u.Quantity = 0 * u.mm

    @classmethod
    def promote(cls, value: 'Translate'):
        return cls(value.x, value.y, value.z)

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



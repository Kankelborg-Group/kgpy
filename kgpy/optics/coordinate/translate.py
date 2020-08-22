import dataclasses
import numpy as np
from astropy import units as u
import kgpy.vector
from . import Decenter

__all__ = ['Translate']


@dataclasses.dataclass
class Translate(Decenter):

    z: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.z,
        )

    def __invert__(self) -> 'Translate':
        return Translate(
            -self.x,
            -self.y,
            -self.z,
        )

    def __add__(self, other: 'Translate'):
        return Translate(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    @property
    def translation_eff(self) -> u.Quantity:
        value = super().translation_eff
        value[kgpy.vector.z] = self.z
        return value

    def copy(self) -> 'Translate':
        return Translate(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy()
        )



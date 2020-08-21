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

    def __call__(
            self,
            value: u.Quantity,
            use_rotations: bool = True,
            use_translations: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        value = super().__call__(value, use_rotations, use_translations, num_extra_dims)
        if use_translations:
            sh = list(self.z.shape)
            sh[~1:~1] = [1] * num_extra_dims
            z = self.z.reshape(sh)
            value[..., 2] += z
        return value

    def copy(self) -> 'Translate':
        return Translate(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy()
        )



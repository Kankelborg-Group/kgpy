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

    def __call__(self, value: u.Quantity, inverse: bool = False, num_extra_dims: int = 0) -> u.Quantity:
        value = super().__call__(value=value, inverse=inverse, num_extra_dims=num_extra_dims)
        sh = list(self.z.shape)
        sh[~1:~1] = [1] * num_extra_dims
        z = self.z.reshape(sh)
        if not inverse:
            value[..., 2] += z
        else:
            value[..., 2] -= z
        return value

    def copy(self) -> 'Translate':
        return type(self)(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy()
        )



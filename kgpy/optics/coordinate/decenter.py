import dataclasses
import numpy as np
from astropy import units as u
import kgpy.mixin

__all__ = ['Decenter']


@dataclasses.dataclass
class Decenter(kgpy.mixin.Broadcastable):

    x: u.Quantity = 0 * u.mm
    y: u.Quantity = 0 * u.mm

    @classmethod
    def promote(cls, value: 'Decenter'):
        return cls(value.x, value.y)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.x,
            self.y,
        )

    def __invert__(self):
        return type(self)(
            -self.x,
            -self.y,
        )

    def __call__(self, value: u.Quantity, inverse: bool = False, num_extra_dims: int = 0) -> u.Quantity:
        value = value.copy()
        x = self.x
        y = self.y
        for i in range:
            pass
        if not inverse:
            value[..., 0] += self.x
            value[..., 1] += self.y
        else:
            value[..., 0] -= self.x
            value[..., 1] -= self.y
        return value


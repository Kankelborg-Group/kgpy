import typing as typ
import dataclasses
import numpy as np
from astropy import units as u
import kgpy.optics
from . import Transform

__all__ = ['Decenter']


@dataclasses.dataclass
class Decenter(Transform):

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

    def __invert__(self) -> 'Decenter':
        return Decenter(
            -self.x,
            -self.y,
        )

    def __call__(
            self,
            value: u.Quantity,
            use_rotations: bool = True,
            use_translations: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        value = value.copy()
        if use_translations:
            sh = list(self.x.shape)
            sh[~1:~1] = [1] * num_extra_dims
            x = self.x.reshape(sh)
            y = self.y.reshape(sh)
            value[..., 0] += x
            value[..., 1] += y
        return value

    def copy(self):
        return Decenter(
            x=self.x.copy(),
            y=self.y.copy(),
        )


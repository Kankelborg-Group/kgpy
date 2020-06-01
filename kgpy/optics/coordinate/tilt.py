import dataclasses
import typing as typ
import numpy as np
from astropy import units as u

import kgpy.matrix
import kgpy.mixin
from kgpy import vector

__all__ = ['Tilt']


@dataclasses.dataclass
class Tilt(kgpy.mixin.Broadcastable):

    x: u.Quantity = 0 * u.deg
    y: u.Quantity = 0 * u.deg
    z: u.Quantity = 0 * u.deg

    @classmethod
    def promote(cls, value: 'Tilt'):
        return cls(value.x, value.y, value.z)

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

    def apply(self, value: u.Quantity, inverse: bool = False, extra_dim: bool = False) -> np.ndarray:
        rot = self.rotation(inverse)
        if extra_dim:
            rot = np.expand_dims(rot, ~2)
        return kgpy.vector.matmul(rot, value)

    def rotation(self, inverse: bool = False) -> np.ndarray:
        x = self.rotation_x(inverse)
        y = self.rotation_y(inverse)
        z = self.rotation_z(inverse)
        if not inverse:
            return kgpy.matrix.mul(kgpy.matrix.mul(x, y), z)
        else:
            return kgpy.matrix.mul(kgpy.matrix.mul(z, y), x)

    def rotation_x(self, inverse: bool = False) -> np.ndarray:
        r = np.zeros(self.shape + (3, 3))
        x = self.x.copy()
        if inverse:
            x *= -1
        r[..., 0, 0] = 1
        r[..., 1, 1] = np.cos(x)
        r[..., 1, 2] = -np.sin(x)
        r[..., 2, 1] = np.sin(x)
        r[..., 2, 2] = np.cos(x)
        return r

    def rotation_y(self, inverse: bool = False) -> np.ndarray:
        r = np.zeros(self.shape + (3, 3))
        y = self.y.copy()
        if inverse:
            y *= -1
        r[..., 0, 0] = np.cos(y)
        r[..., 0, 2] = np.sin(y)
        r[..., 1, 1] = 1
        r[..., 2, 0] = -np.sin(y)
        r[..., 2, 2] = np.cos(y)
        return r

    def rotation_z(self, inverse: bool = False) -> np.ndarray:
        r = np.zeros(self.shape + (3, 3))
        z = self.z.copy()
        if inverse:
            z *= -1
        r[..., 0, 0] = np.cos(z)
        r[..., 0, 1] = -np.sin(z)
        r[..., 1, 0] = np.sin(z)
        r[..., 1, 1] = np.cos(z)
        r[..., 2, 2] = 1
        return r


import dataclasses
import typing as typ
import numpy as np
from astropy import units as u
from ... import mixin

__all__ = ['Tilt']


@dataclasses.dataclass
class Tilt(mixin.Broadcastable):

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

    def apply(self, value: u.Quantity, inverse: bool = False) -> np.ndarray:
        rot = self.rotation(inverse)
        new_value = np.empty_like(value)
        new_value[0] = rot[0, 0] * value[..., 0] + rot[0, 1] * value[..., 1] + rot[0, 2] * value[..., 2]
        new_value[1] = rot[1, 0] * value[..., 0] + rot[1, 1] * value[..., 1] + rot[1, 2] * value[..., 2]
        new_value[2] = rot[2, 0] * value[..., 0] + rot[2, 1] * value[..., 1] + rot[2, 2] * value[..., 2]
        return new_value

    def rotation(self, inverse: bool = False) -> np.ndarray:
        if not inverse:
            return self.rotation_x @ self.rotation_y @ self.rotation_z
        else:
            return self.rotation_z @ self.rotation_y @ self.rotation_x

    @property
    def rotation_x(self) -> np.ndarray:
        return np.array([
            [1, 0, 0],
            [0, np.cos(self.x), -np.sin(self.x)],
            [0, np.sin(self.x), np.cos(self.x)],
        ])

    @property
    def rotation_y(self) -> np.ndarray:
        return np.array([
            [np.cos(self.y), 0, np.sin(self.y)],
            [0, 1, 0],
            [-np.sin(self.y), 0, np.cos(self.y)],
        ])

    @property
    def rotation_z(self) -> np.ndarray:
        return np.array([
            [np.cos(self.z), -np.sin(self.z), 0],
            [np.sin(self.z), np.cos(self.z), 0],
            [0, 0, 1],
        ])


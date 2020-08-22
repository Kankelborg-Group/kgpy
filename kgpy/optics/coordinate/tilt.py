import typing as typ
import abc
import dataclasses
import numpy as np
from astropy import units as u
import kgpy.matrix
import kgpy.mixin
from kgpy import vector
from . import Transform, TransformList

__all__ = ['TiltX', 'TiltY', 'TiltZ', 'TiltXYZ']


@dataclasses.dataclass
class TiltAboutAxis(Transform, abc.ABC):
    angle: u.Quantity = 0 * u.deg

    @property
    def config_broadcast(self):
        return np.broadcast(super().config_broadcast, self.angle)

    def __eq__(self, other: 'TiltAboutAxis') -> bool:
        return np.array(self.angle == other.angle).all()

    def __invert__(self) -> 'TiltAboutAxis':
        return type(self)(angle=-self.angle)

    @property
    def translation_eff(self) -> u.Quantity:
        return super().translation_eff

    def copy(self) -> 'TiltAboutAxis':
        return type(self)(
            angle=self.angle.copy(),
        )


class TiltX(TiltAboutAxis):

    @property
    def rotation_eff(self) -> u.Quantity:
        r = np.zeros(self.shape + (3, 3)) << u.dimensionless_unscaled
        cos_x, sin_x = np.cos(-self.angle), np.sin(-self.angle)
        r[..., 0, 0] = 1
        r[..., 1, 1] = cos_x
        r[..., 1, 2] = sin_x
        r[..., 2, 1] = -sin_x
        r[..., 2, 2] = cos_x
        return r


class TiltY(TiltAboutAxis):

    @property
    def rotation_eff(self) -> u.Quantity:
        r = np.zeros(self.shape + (3, 3)) << u.dimensionless_unscaled
        cos_y, sin_y = np.cos(-self.angle), np.sin(-self.angle)
        r[..., 0, 0] = cos_y
        r[..., 0, 2] = -sin_y
        r[..., 1, 1] = 1
        r[..., 2, 0] = sin_y
        r[..., 2, 2] = cos_y
        return r


class TiltZ(TiltAboutAxis):

    @property
    def rotation_eff(self) -> u.Quantity:
        r = np.zeros(self.shape + (3, 3)) << u.dimensionless_unscaled
        cos_z, sin_z = np.cos(-self.angle), np.sin(-self.angle)
        r[..., 0, 0] = cos_z
        r[..., 0, 1] = sin_z
        r[..., 1, 0] = -sin_z
        r[..., 1, 1] = cos_z
        r[..., 2, 2] = 1
        return r


@dataclasses.dataclass
class TiltXYZ(Transform):

    x: u.Quantity = 0 * u.deg
    y: u.Quantity = 0 * u.deg
    z: u.Quantity = 0 * u.deg

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.x,
            self.y,
            self.z,
        )

    def __eq__(self, other: 'TiltXYZ'):
        out = True
        out &= np.array(self.x == other.x).all()
        out &= np.array(self.y == other.y).all()
        out &= np.array(self.z == other.z).all()
        return out

    @property
    def _transform(self) -> TransformList:
        return TransformList([TiltX(self.x), TiltY(self.y), TiltZ(self.z)])

    def __invert__(self) -> 'TransformList':
        return self._transform.__invert__()

    def __call__(
            self,
            value: u.Quantity,
            use_rotations: bool = True,
            use_translations: bool = True,
            num_extra_dims: int = 0,
    ) -> np.ndarray:
        return self._transform(value, use_rotations, use_translations, num_extra_dims)

    def copy(self) -> 'TiltXYZ':
        return TiltXYZ(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
        )

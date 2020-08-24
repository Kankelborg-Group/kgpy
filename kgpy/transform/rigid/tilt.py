import typing as typ
import abc
import dataclasses
import numpy as np
from astropy import units as u
from . import Transform

__all__ = ['TiltX', 'TiltY', 'TiltZ']


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
    def translation_eff(self) -> None:
        return None

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
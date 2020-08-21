import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.vector
from . import Aperture, decenterable, obscurable

__all__ = ['Circular']


@dataclasses.dataclass
class Circular(decenterable.Decenterable, obscurable.Obscurable, Aperture):

    radius: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius,
        )

    @property
    def min(self) -> u.Quantity:
        return -self.max

    @property
    def max(self) -> u.Quantity:
        return kgpy.vector.from_components(self.radius, self.radius)

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        x = points[..., 0]
        y = points[..., 1]
        r = np.sqrt(np.square(x) + np.square(y))
        is_inside = r < self.radius
        if not self.is_obscuration:
            return is_inside
        else:
            return ~is_inside

    @property
    def wire(self) -> u.Quantity:

        a = np.linspace(0 * u.deg, 360 * u.deg, num=self.num_samples)
        r = np.expand_dims(self.radius.copy(), ~0)

        x = r * np.cos(a)
        y = r * np.sin(a)
        z = np.broadcast_to(0, x.shape)

        return np.stack([x, y, z], axis=~0)

    def copy(self) -> 'Circular':
        return Circular(
            num_samples=self.num_samples,
            is_active=self.is_active,
            is_test_stop=self.is_test_stop,
            is_obscuration=self.is_obscuration,
            decenter=self.decenter.copy(),
            radius=self.radius.copy(),
        )

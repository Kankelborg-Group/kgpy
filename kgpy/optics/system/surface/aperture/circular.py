import dataclasses
import numpy as np
import astropy.units as u

from . import Aperture, decenterable, obscurable

__all__ = ['Circular']


@dataclasses.dataclass
class Circular(obscurable.Obscurable, decenterable.Decenterable, Aperture):

    inner_radius: u.Quantity = 0 * u.mm
    outer_radius: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.inner_radius,
            self.outer_radius,
        )

    @property
    def points(self) -> u.Quantity:
        a = np.linspace(0 * u.deg, 360 * u.deg, num=100)

        x = u.Quantity([np.cos(a), np.sin(a)])
        x = np.moveaxis(x, 0, ~0)

        r0, r1 = np.broadcast_arrays(self.outer_radius, self.inner_radius)

        r0 = np.reshape(r0, r0.shape + (1, 1, 1))
        r1 = np.reshape(r1, r1.shape + (1, 1, 1))

        return np.stack([r0 * x, r1 * x], axis=~2)

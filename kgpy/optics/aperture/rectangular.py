import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.vector
from . import polygon

__all__ = ['Rectangular']


@dataclasses.dataclass
class Rectangular(polygon.Polygon):

    half_width_x: u.Quantity = 0 * u.mm
    half_width_y: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.half_width_x,
            self.half_width_y,
        )

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        x = points[kgpy.vector.x]
        y = points[kgpy.vector.y]
        m1 = x <= self.half_width_x
        m2 = x >= -self.half_width_x
        m3 = y <= self.half_width_y
        m4 = y >= -self.half_width_y
        is_inside = m1 & m2 & m3 & m4
        if not self.is_obscuration:
            return is_inside
        else:
            return ~is_inside

    @property
    def min(self) -> u.Quantity:
        return -self.max

    @property
    def max(self) -> u.Quantity:
        return kgpy.vector.from_components(self.half_width_x, self.half_width_y)

    @property
    def vertices(self) -> u.Quantity:

        wx, wy = np.broadcast_arrays(self.half_width_x, self.half_width_y, subok=True)

        x = np.stack([wx, wx, -wx, -wx], axis=~0)
        y = np.stack([wy, -wy, -wy, wy], axis=~0)

        return kgpy.vector.from_components(x, y)

    # @property
    # def wire(self) -> u.Quantity:
    #
    #     wx, wy = np.broadcast_arrays(self.half_width_x, self.half_width_y, subok=True)
    #
    #     rx = np.linspace(-wx, wx, self.num_samples, axis=~0)
    #     ry = np.linspace(-wy, wy, self.num_samples, axis=~0)
    #
    #     wx = np.expand_dims(wx, ~0)
    #     wy = np.expand_dims(wy, ~0)
    #
    #     wx, rx = np.broadcast_arrays(wx, rx, subok=True)
    #     wy, ry = np.broadcast_arrays(wy, ry, subok=True)
    #
    #     x = np.stack([rx, wx, rx[::-1], -wx])
    #     y = np.stack([wy, ry[::-1], -wy, ry])
    #
    #     return kgpy.vector.from_components(x, y)

    def copy(self) -> 'Rectangular':
        return Rectangular(
            num_samples=self.num_samples,
            is_active=self.is_active,
            is_test_stop=self.is_test_stop,
            is_obscuration=self.is_obscuration,
            decenter=self.decenter.copy(),
            half_width_x=self.half_width_x.copy(),
            half_width_y=self.half_width_y.copy(),
        )

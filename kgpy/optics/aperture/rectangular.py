import dataclasses
import numpy as np
import astropy.units as u
from . import Aperture, decenterable, obscurable

__all__ = ['Rectangular']


@dataclasses.dataclass
class Rectangular(decenterable.Decenterable, obscurable.Obscurable, Aperture):

    half_width_x: u.Quantity = 0 * u.mm
    half_width_y: u.Quantity = 0 * u.mm

    def to_zemax(self) -> 'Rectangular':
        from kgpy.optics import zemax
        return zemax.system.surface.aperture.Rectangular(
            is_obscuration=self.is_obscuration,
            decenter=self.decenter,
            half_width_x=self.half_width_x,
            half_width_y=self.half_width_y,
        )

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.half_width_x,
            self.half_width_y,
        )

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        x = points[..., 0]
        y = points[..., 1]
        m1 = x < self.half_width_x
        m2 = x > -self.half_width_x
        m3 = y < self.half_width_y
        m4 = y > -self.half_width_y
        return m1 & m2 & m3 & m4

    @property
    def points(self) -> u.Quantity:

        wx, wy = np.broadcast_arrays(self.half_width_x, self.half_width_y, subok=True)

        rx = np.linspace(-wx, wx, self.points_per_segment, axis=~0)
        ry = np.linspace(-wy, wy, self.points_per_segment, axis=~0)

        wx = np.expand_dims(wx, ~0)
        wy = np.expand_dims(wy, ~0)

        wx, rx = np.broadcast_arrays(wx, rx, subok=True)
        wy, ry = np.broadcast_arrays(wy, ry, subok=True)

        x = np.concatenate([rx, wx, rx[::-1], -wx], axis=~0)
        y = np.concatenate([wy, ry[::-1], -wy, ry], axis=~0)
        z = np.broadcast_to(0, x.shape)

        return np.stack([x, y, z], axis=~0)

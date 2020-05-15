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

    def check_points(self, points: u.Quantity) -> np.ndarray:
        m1 = points[0, ...] < self.half_width_x
        m2 = points[0, ...] > -self.half_width_x
        m3 = points[1, ...] < self.half_width_y
        m4 = points[1, ...] > -self.half_width_y
        return m1 & m2 & m3 & m4

    @property
    def points(self) -> u.Quantity:

        wx, wy = np.broadcast_arrays(self.half_width_x, self.half_width_y, subok=True)

        x = np.stack([wx, wx, -wx, -wx], axis=~0)
        y = np.stack([wy, -wy, wy, -wy], axis=~0)

        return np.stack([x, y], axis=~0)

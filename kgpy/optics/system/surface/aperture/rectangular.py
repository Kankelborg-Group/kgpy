import dataclasses
import numpy as np
import astropy.units as u

from . import Aperture, decenterable, obscurable

__all__ = ['Rectangular']


@dataclasses.dataclass
class Rectangular(decenterable.Decenterable, obscurable.Obscurable, Aperture):

    half_width_x: u.Quantity = 0 * u.mm
    half_width_y: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.half_width_x,
            self.half_width_y,
        )

    @property
    def points(self) -> u.Quantity:

        wx, wy = np.broadcast_arrays(self.half_width_x, self.half_width_y, subok=True)

        x = np.stack([wx, wx, -wx, -wx], axis=~0)
        y = np.stack([wy, -wy, wy, -wy], axis=~0)

        return np.stack([x, y], axis=~0)

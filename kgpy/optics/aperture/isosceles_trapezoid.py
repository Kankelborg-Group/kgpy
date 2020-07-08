import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from . import Polygon

__all__ = ['IsoscelesTrapezoid']


@dataclasses.dataclass
class IsoscelesTrapezoid(Polygon):

    inner_radius: u.Quantity = 0 * u.mm
    outer_radius: u.Quantity = 0 * u.mm
    wedge_angle: u.Quantity = 0 * u.deg

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.half_width_x,
            self.half_width_y,
        )
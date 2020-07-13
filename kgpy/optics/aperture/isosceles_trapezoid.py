import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.vector
from . import Polygon

__all__ = ['IsoscelesTrapezoid']


@dataclasses.dataclass
class IsoscelesTrapezoid(Polygon):

    inner_radius: u.Quantity = 0 * u.mm
    outer_radius: u.Quantity = 0 * u.mm
    wedge_half_angle: u.Quantity = 0 * u.deg
    
    @property
    def vertices(self) -> u.Quantity:
        m = np.tan(self.wedge_half_angle)
        bottom_x, bottom_y = self.inner_radius + self.decenter.x, m * self.inner_radius + self.decenter.y
        top_x, top_y = self.outer_radius + self.decenter.x, m * self.outer_radius + self.decenter.y
        v_x = np.stack([top_x, -top_x, bottom_x, -bottom_x], axis=~0)
        v_y = np.stack([top_y, top_y, bottom_y, bottom_y], axis=~0)
        return kgpy.vector.from_components(v_x, v_y)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.half_width_x,
            self.half_width_y,
        )
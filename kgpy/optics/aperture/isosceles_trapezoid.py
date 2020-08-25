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
        left_x, left_y = self.inner_radius + self.decenter.x, m * self.inner_radius + self.decenter.y
        right_x, right_y = self.outer_radius + self.decenter.x, m * self.outer_radius + self.decenter.y
        v_x = np.stack([left_x, right_x, right_x, left_x], axis=~0)
        v_y = np.stack([left_y, right_y, -right_y, -left_y], axis=~0)
        return kgpy.vector.from_components(v_x, v_y)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.inner_radius,
            self.outer_radius,
            self.wedge_half_angle,
        )

    def copy(self) -> 'IsoscelesTrapezoid':
        other = super().copy()      # type: IsoscelesTrapezoid
        other.inner_radius = self.inner_radius.copy()
        other.outer_radius = self.outer_radius.copy()
        other.wedge_half_angle = self.wedge_half_angle.copy()
        return other

import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.vector
from . import Polygon

__all__ = ['IsoscelesTrapezoid']


@dataclasses.dataclass
class IsoscelesTrapezoid(Polygon):
    apex_offset: u.Quantity = 0 * u.mm
    half_width_left: u.Quantity = 0 * u.mm
    half_width_right: u.Quantity = 0 * u.mm
    wedge_half_angle: u.Quantity = 0 * u.deg

    @property
    def vertices(self) -> u.Quantity:
        m = np.tan(self.wedge_half_angle)
        # inner_radius = self.half_width_left +
        # inner_radius = self.apex_offset - self.half_width_left
        # outer_radius = self.apex_offset + self.half_width_right
        left_x, left_y = -self.half_width_left, -m * (self.apex_offset + self.half_width_left)
        right_x, right_y = self.half_width_right, -m * (self.apex_offset - self.half_width_right)
        v_x = np.stack([left_x, right_x, right_x, left_x], axis=~0)
        v_y = np.stack([left_y, right_y, -right_y, -left_y], axis=~0)
        return kgpy.vector.from_components(v_x, v_y)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.apex_offset)
        out = np.broadcast(out, self.half_width_left)
        out = np.broadcast(out, self.half_width_right)
        out = np.broadcast(out, self.wedge_half_angle)
        return out

    def copy(self) -> 'IsoscelesTrapezoid':
        other = super().copy()      # type: IsoscelesTrapezoid
        other.apex_offset = self.apex_offset.copy()
        other.half_width_left = self.half_width_left.copy()
        other.half_width_right = self.half_width_right.copy()
        other.wedge_half_angle = self.wedge_half_angle.copy()
        return other

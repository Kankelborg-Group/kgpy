import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.vector
from . import Polygon

__all__ = ['AsymmetricRectangular']


@dataclasses.dataclass
class AsymmetricRectangular(Polygon):

    width_x_neg: u.Quantity = 0 * u.mm
    width_x_pos: u.Quantity = 0 * u.mm
    width_y_neg: u.Quantity = 0 * u.mm
    width_y_pos: u.Quantity = 0 * u.mm

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.width_x_neg)
        out = np.broadcast(out, self.width_x_pos)
        out = np.broadcast(out, self.width_y_neg)
        out = np.broadcast(out, self.width_y_pos)
        return out

    @property
    def vertices(self) -> u.Quantity:
        v_x = np.stack([self.width_x_pos, self.width_x_neg, self.width_x_neg, self.width_x_pos])
        v_y = np.stack([self.width_y_pos, self.width_y_pos, self.width_y_neg, self.width_y_neg])
        return kgpy.vector.from_components(v_x, v_y)

    def copy(self) -> 'AsymmetricRectangular':
        other = super().copy()  # type: AsymmetricRectangular
        other.width_x_neg = self.width_x_neg.copy()
        other.width_x_pos = self.width_x_pos.copy()
        other.width_y_neg = self.width_y_neg.copy()
        other.width_y_pos = self.width_y_pos.copy()
        return other


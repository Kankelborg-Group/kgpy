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
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.width_x_neg,
            self.width_x_pos,
            self.width_y_neg,
            self.width_y_pos,
        )

    @property
    def vertices(self) -> u.Quantity:
        v_x = np.stack([self.width_x_pos, self.width_x_neg, self.width_x_neg, self.width_x_pos])
        v_y = np.stack([self.width_y_pos, self.width_y_pos, self.width_y_neg, self.width_y_neg])
        return kgpy.vector.from_components(v_x, v_y)

    def copy(self) -> 'AsymmetricRectangular':
        return AsymmetricRectangular(
            num_samples=self.num_samples,
            is_active=self.is_active,
            is_test_stop=self.is_test_stop,
            is_obscuration=self.is_obscuration,
            decenter=self.decenter.copy(),
            width_x_neg=self.width_x_neg.copy(),
            width_x_pos=self.width_x_pos.copy(),
            width_y_neg=self.width_y_neg.copy(),
            width_y_pos=self.width_y_pos.copy(),
        )


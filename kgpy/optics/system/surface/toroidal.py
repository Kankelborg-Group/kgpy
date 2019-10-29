import dataclasses
import numpy as np
import astropy.units as u

from . import Standard

__all__ = ['Toroidal']


@dataclasses.dataclass
class Toroidal(Standard):

    radius_of_rotation: u.Quantity = 0 * u.mm

    @property
    def broadcasted_attrs(self):
        return np.broadcast(
            super().broadcasted_attrs,
            self.radius_of_rotation,
        )
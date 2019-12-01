import dataclasses
import numpy as np
import astropy.units as u

from . import Standard

__all__ = ['Toroidal']


@dataclasses.dataclass
class Toroidal(Standard):

    radius_of_rotation: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.mm)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius_of_rotation,
        )
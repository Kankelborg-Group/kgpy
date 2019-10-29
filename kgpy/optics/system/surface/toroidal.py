import dataclasses
import astropy.units as u

from . import Standard

__all__ = ['Toroidal']


@dataclasses.dataclass
class Toroidal(Standard):

    radius_of_rotation: u.Quantity = 0 * u.mm

    @property
    def broadcastable_attrs(self):
        return super().broadcastable_attrs + [
            self.radius_of_rotation,
        ]
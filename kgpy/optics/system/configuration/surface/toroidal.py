
import astropy.units as u

from . import Standard

__all__ = ['Toroidal']


class Toroidal(Standard):

    def __init__(self, radius_of_rotation: u.Quantity = None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if radius_of_rotation is None:
            radius_of_rotation = 0 * u.m

        self._radius_of_rotation = radius_of_rotation

    @property
    def radius_of_rotation(self) -> u.Quantity:
        return self._radius_of_rotation

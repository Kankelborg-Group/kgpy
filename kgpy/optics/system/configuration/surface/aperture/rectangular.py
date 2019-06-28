
import astropy.units as u

from . import Aperture

__all__ = ['Rectangular']


class Rectangular(Aperture):

    def __init__(self):

        super().__init__()

        self.half_width_x = 0 * u.m
        self.half_width_y = 0 * u.m

    @property
    def half_width_x(self) -> u.Quantity:
        return self._half_width_x

    @half_width_x.setter
    def half_width_x(self, value: u.Quantity):
        self._half_width_x = value

    @property
    def half_width_y(self) -> u.Quantity:
        return self._half_width_y

    @half_width_y.setter
    def half_width_y(self, value: u.Quantity):
        self._half_width_y = value

    @property
    def points(self):
        p = u.Quantity([
            u.Quantity([self.half_width_x, self.half_width_y]),
            u.Quantity([self.half_width_x, -self.half_width_y]),
            u.Quantity([-self.half_width_x, -self.half_width_y]),
            u.Quantity([-self.half_width_x, self.half_width_y]),
        ])

        return p

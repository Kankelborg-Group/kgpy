
import astropy.units as u

from . import Aperture

__all__ = ['Rectangular']


class Rectangular(Aperture):

    def __init__(self, *args,
                 is_obscuration: bool = False,
                 decenter_x: u.Quantity = None,
                 decenter_y: u.Quantity = None,
                 half_width_x: u.Quantity = None,
                 half_width_y: u.Quantity = None,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)

        if decenter_x is None:
            decenter_x = 0 * u.m
        if decenter_y is None:
            decenter_y = 0 * u.m
        if half_width_x is None:
            half_width_x = 0 * u.m
        if half_width_y is None:
            half_width_y = 0 * u.m

        self._is_obscuration = is_obscuration
        self._decenter_x = decenter_x
        self._decenter_y = decenter_y
        self._half_width_x = half_width_x
        self._half_width_y = half_width_y

        self._points = u.Quantity([
            [self.half_width_x, self.half_width_y],
            [self.half_width_x, -self.half_width_y],
            [-self.half_width_x, -self.half_width_y],
            [-self.half_width_x, self.half_width_y],
        ])

    @property
    def is_obscuration(self) -> bool:
        return self._is_obscuration

    @property
    def decenter_x(self) -> u.Quantity:
        return self._decenter_x

    @property
    def decenter_y(self) -> u.Quantity:
        return self._decenter_y

    @property
    def half_width_x(self) -> u.Quantity:
        return self._half_width_x

    @property
    def half_width_y(self) -> u.Quantity:
        return self._half_width_y

    @property
    def points(self) -> u.Quantity:
        return self._points

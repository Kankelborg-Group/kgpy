
import astropy.units as u

__all__ = []


class Field:

    def __init__(self, x: u.Quantity, y: u.Quantity):

        self.x = x
        self.y = y

        self.van = 0.0 * u.rad
        self.vdx = 0.0 * u.dimensionless_unscaled
        self.vdy = 0.0 * u.dimensionless_unscaled
        self.vcx = 0.0 * u.dimensionless_unscaled
        self.vcy = 0.0 * u.dimensionless_unscaled

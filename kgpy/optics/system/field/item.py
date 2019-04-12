
import astropy.units as u

__all__ = []


class Item:

    def __init__(self, x: u.Quantity, y: u.Quantity):

        self.x = x
        self.y = y

        self.van = 0.0 * u.rad
        self.vdx = 0.0
        self.vdy = 0.0
        self.vcx = 0.0
        self.vcy = 0.0


import astropy.units as u

__all__ = []


class Field:

    def __init__(self):

        self.x = 0.0 * u.rad
        self.y = 0.0 * u.rad
        self.weight = 0.0 * u.dimensionless_unscaled

        self.van = 0.0 * u.rad
        self.vdx = 0.0 * u.dimensionless_unscaled
        self.vdy = 0.0 * u.dimensionless_unscaled
        self.vcx = 0.0 * u.dimensionless_unscaled
        self.vcy = 0.0 * u.dimensionless_unscaled

    @property
    def x(self) -> u.Quantity:
        return self._x

    @x.setter
    def x(self, value: u.Quantity):
        self._x = value

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        
    @property
    def weight(self) -> u.Quantity:
        return self._weight
    
    @weight.setter
    def weight(self, value: u.Quantity):
        self._weight = value

    @property
    def vdx(self) -> u.Quantity:
        return self._vdx
    
    @vdx.setter
    def vdx(self, value: u.Quantity):
        self._vdx = value

    @property
    def vdy(self) -> u.Quantity:
        return self._vdy

    @vdy.setter
    def vdy(self, value: u.Quantity):
        self._vdy = value

    @property
    def vcx(self) -> u.Quantity:
        return self._vcx

    @vcx.setter
    def vcx(self, value: u.Quantity):
        self._vcx = value
        
    @property
    def vcy(self) -> u.Quantity:
        return self._vcy

    @vcy.setter
    def vcy(self, value: u.Quantity):
        self._vcy = value
        
    @property
    def van(self: u.Quantity):
        return self._van

    @van.setter
    def van(self, value: u.Quantity):
        self._van = value

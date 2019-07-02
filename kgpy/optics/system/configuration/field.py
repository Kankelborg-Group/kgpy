
import astropy.units as u

__all__ = []


class Field:

    def __init__(self,
                 x: u.Quantity = None,
                 y: u.Quantity = None,
                 weight: u.Quantity = None,
                 vdx: u.Quantity = None,
                 vdy: u.Quantity = None,
                 vcx: u.Quantity = None,
                 vcy: u.Quantity = None,
                 van: u.Quantity = None
                 ):

        if x is None:
            x = 0.0 * u.rad
        if y is None:
            y = 0.0 * u.rad
        if weight is None:
            weight = 1.0 * u.dimensionless_unscaled
        if vdx is None:
            vdx = 0.0 * u.dimensionless_unscaled
        if vdy is None:
            vdy = 0.0 * u.dimensionless_unscaled
        if vcx is None:
            vcx = 0.0 * u.dimensionless_unscaled
        if vcy is None:
            vcy = 0.0 * u.dimensionless_unscaled
        if van is None:
            van = 0.0 * u.rad

        self._x = x
        self._y = y
        self._weight = weight
        self._vdx = vdx
        self._vdy = vdy
        self._vcx = vcx
        self._vcy = vcy
        self._van = van

    @property
    def x(self) -> u.Quantity:
        return self._x

    @property
    def y(self) -> u.Quantity:
        return self._y
        
    @property
    def weight(self) -> u.Quantity:
        return self._weight

    @property
    def vdx(self) -> u.Quantity:
        return self._vdx

    @property
    def vdy(self) -> u.Quantity:
        return self._vdy

    @property
    def vcx(self) -> u.Quantity:
        return self._vcx
        
    @property
    def vcy(self) -> u.Quantity:
        return self._vcy
        
    @property
    def van(self: u.Quantity):
        return self._van

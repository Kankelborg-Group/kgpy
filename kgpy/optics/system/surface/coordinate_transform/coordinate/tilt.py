import dataclasses
import typing as typ
import astropy.units as u
from kgpy.component import Component
from .... import surface
from .. import coordinate

__all__ = ['Tilt']


@dataclasses.dataclass
class Tilt(Component[coordinate.Transform], surface.coordinate.Translate):

    def _update(self) -> typ.NoReturn:
        self.x = self.x
        self.y = self.y
        self.z = self.z

    @property
    def x(self) -> u.Quantity:
        return self._x

    @x.setter
    def x(self, value: u.Quantity):
        self._x = value
        try:
            self.composite.composite._cb_tilt.tilt.x = value
            self.composite.composite._cb_translate.tilt.x = 0
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self.composite.composite._cb_tilt.tilt.y = value
            self.composite.composite._cb_translate.tilt.y = 0
        except AttributeError:
            pass

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self.composite.composite._cb_tilt.tilt.z = value
            self.composite.composite._cb_translate.tilt.z = 0
        except AttributeError:
            pass

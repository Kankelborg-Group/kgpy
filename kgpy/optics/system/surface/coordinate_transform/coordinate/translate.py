import dataclasses
import typing as typ
import astropy.units as u
from kgpy.component import Component
from .... import surface
from .. import coordinate

__all__ = ['Translate']


@dataclasses.dataclass
class Translate(Component[coordinate.Transform], surface.coordinate.Translate):

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
            self.composite.composite._cb_translate.decenter.x = value
            self.composite.composite._cb_tilt.decenter.x = 0
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self.composite.composite._cb_translate.decenter.y = value
            self.composite.composite._cb_tilt.decenter.y = value
        except AttributeError:
            pass

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self.composite.composite._cb_translate.thickness = value
            self.composite.composite._cb_tilt.thickness = 0
        except AttributeError:
            pass

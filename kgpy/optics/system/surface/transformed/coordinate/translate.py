import dataclasses
import typing as typ
import astropy.units as u
from kgpy.component import Component
from .... import surface
from .. import coordinate

__all__ = ['Translate']


@dataclasses.dataclass
class Translate(Component['coordinate.Transform'], surface.coordinate.Translate):

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
            self._composite._ct_before.translate.x = value
            self._composite._ct_after.translate.x = -value
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self._composite._ct_before.translate.y = value
            self._composite._ct_after.translate.y = -value
        except AttributeError:
            pass

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self._composite._ct_before.translate.z = value
            self._composite._ct_after.translate.z = -value
        except AttributeError:
            pass

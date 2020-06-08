import dataclasses
import typing as typ
import astropy.units as u
import kgpy
from .... import coordinate as base_coordinate
from .. import coordinate

__all__ = ['Translate']


@dataclasses.dataclass
class Translate(kgpy.Component['coordinate.Transform'], base_coordinate.Translate):

    def _update(self) -> typ.NoReturn:
        super()._update()
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
            self._composite._ct_before.transform.translate.x = value
            self._composite._ct_after.transform.translate.x = -value
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self._composite._ct_before.transform.translate.y = value
            self._composite._ct_after.transform.translate.y = -value
        except AttributeError:
            pass

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self._composite._ct_before.transform.translate.z = value
            self._composite._ct_after.transform.translate.z = -value
        except AttributeError:
            pass

import dataclasses
import typing as typ
import astropy.units as u
import kgpy
import kgpy.optics.coordinate
from .... import surface
from .. import coordinate

__all__ = ['Translate']


@dataclasses.dataclass
class Translate(kgpy.Component['coordinate.Transform'], kgpy.optics.coordinate.Translate):

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
            self._composite._cb_translate.transform.decenter.x = value
            self._composite._cb_tilt.transform.decenter.x = 0
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self._composite._cb_translate.transform.decenter.y = value
            self._composite._cb_tilt.transform.decenter.y = 0
        except AttributeError:
            pass

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self._composite._cb_translate.thickness = value
            self._composite._cb_tilt.thickness = 0
        except AttributeError:
            pass

import dataclasses
import typing as typ
import astropy.units as u
from kgpy.component import Component
from .... import surface
from .. import coordinate

__all__ = ['Tilt']


@dataclasses.dataclass
class Tilt(Component['coordinate.Transform'], surface.coordinate.Tilt):

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
            self._composite._ct_before.transform.tilt.x = value
            self._composite._ct_after.transform.tilt.x = -value
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self._composite._ct_before.transform.tilt.y = value
            self._composite._ct_after.transform.tilt.y = -value
        except AttributeError:
            pass

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self._composite._ct_before.transform.tilt.z = value
            self._composite._ct_after.transform.tilt.z = -value
        except AttributeError:
            pass
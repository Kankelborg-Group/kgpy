import dataclasses
import typing as typ
from kgpy.component import Component
from .... import surface
from .. import coordinate, coordinate_transform

__all__ = ['Transform']


@dataclasses.dataclass
class Base:

    tilt: coordinate.Tilt = dataclasses.field(default_factory=lambda: coordinate.Tilt())
    translate: coordinate.Translate = dataclasses.field(default_factory=lambda: coordinate.Translate())


@dataclasses.dataclass
class Transform(Component[coordinate_transform.CoordinateTransform], Base, surface.coordinate.Transform):

    def _update(self) -> typ.NoReturn:
        self.tilt_first = self.tilt_first

    @property
    def tilt(self) -> coordinate.Tilt:
        return self._tilt

    @tilt.setter
    def tilt(self, value: coordinate.Tilt):
        value._composite = self
        self._tilt = value

    @property
    def translate(self) -> coordinate.Translate:
        return self._translate

    @translate.setter
    def translate(self, value: coordinate.Translate):
        value._composite = self
        self._translate = value

    @property
    def tilt_first(self) -> bool:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: bool):
        self._tilt_first = value
        try:
            self._composite._cb1.tilt_first = value
            self._composite._cb2.tilt_first = value
        except AttributeError:
            pass
        self.tilt = self.tilt
        self.translate = self.translate

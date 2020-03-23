import dataclasses
import typing as typ
from kgpy.optics.system.surface import coordinate
from ... import Child
from ..surface import SurfaceT
from . import Tilt, Translate, TiltFirst

__all__ = ['Transform']


@dataclasses.dataclass
class Base:

    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    translate: Translate = dataclasses.field(default_factory=lambda: Translate())
    tilt_first: TiltFirst = dataclasses.field(default_factory=lambda: TiltFirst())


@dataclasses.dataclass
class Transform(Child[SurfaceT], coordinate.Transform, typ.Generic[SurfaceT], ):

    def _update(self) -> None:
        self.tilt = self.tilt
        self.translate = self.translate
        self.tilt_first = self.tilt_first

    @property
    def tilt(self) -> Tilt:
        return self._tilt

    @tilt.setter
    def tilt(self, value: Tilt):
        value.tilt_decenter = self
        self._tilt = value

    @property
    def translate(self) -> Translate:
        return self._translate

    @translate.setter
    def translate(self, value: Translate):
        value.tilt_decenter = self
        self._translate = value

    @property
    def tilt_first(self) -> TiltFirst:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: TiltFirst):
        value.parent = self
        self._tilt_first = value

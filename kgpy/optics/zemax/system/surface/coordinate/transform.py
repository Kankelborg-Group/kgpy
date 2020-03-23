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
    def tilt(self) -> Tilt['Transform[SurfaceT]']:
        return self._tilt

    @tilt.setter
    def tilt(self, value: Tilt['Transform[SurfaceT]']):
        self._link(value)
        self._tilt = value

    @property
    def translate(self) -> Translate['Transform[SurfaceT]']:
        return self._translate

    @translate.setter
    def translate(self, value: Translate['Transform[SurfaceT]']):
        self._link(value)
        self._translate = value

    @property
    def tilt_first(self) -> TiltFirst['Transform[SurfaceT]']:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: TiltFirst['Transform[SurfaceT]']):
        self._link(value)
        self._tilt_first = value

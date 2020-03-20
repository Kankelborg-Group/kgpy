import dataclasses
import typing as typ
from kgpy.optics.system import coordinate
from ..descendants import Child, SurfaceT
from . import Tilt, Translate, TiltFirst

__all__ = ['Transform']


class Transform(typ.Generic[SurfaceT], Child[SurfaceT], coordinate.Transform):

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
        value.tilt_decenter = self
        self._tilt_first = value

    @property
    def parent(self) -> SurfaceT:
        return self._parent

    @parent.setter
    def parent(self, value: SurfaceT):
        self._parent = value
        self._update()

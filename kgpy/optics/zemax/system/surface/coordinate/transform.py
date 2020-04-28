import dataclasses
import typing as typ
from kgpy.component import Component
from kgpy.optics.system.surface import coordinate
from ... import surface
from . import Tilt, Translate

__all__ = ['Transform']

SurfaceT = typ.TypeVar('SurfaceT', bound='surface.Surface')


@dataclasses.dataclass
class Base:

    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    translate: Translate = dataclasses.field(default_factory=lambda: Translate())


@dataclasses.dataclass
class Transform(Component[SurfaceT], coordinate.Transform, typ.Generic[SurfaceT], ):

    def _update(self) -> None:
        super()._update()
        self.tilt = self.tilt
        self.translate = self.translate
        self.tilt_first = self.tilt_first

    @property
    def tilt(self) -> Tilt['Transform[SurfaceT]']:
        return self._tilt

    @tilt.setter
    def tilt(self, value: Tilt['Transform[SurfaceT]']):
        value._composite = self
        self._tilt = value

    @property
    def translate(self) -> Translate['Transform[SurfaceT]']:
        return self._translate

    @translate.setter
    def translate(self, value: Translate['Transform[SurfaceT]']):
        value._composite = self
        self._translate = value

    @property
    def tilt_first(self) -> bool:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: bool):
        value._composite = self
        self._tilt_first = value

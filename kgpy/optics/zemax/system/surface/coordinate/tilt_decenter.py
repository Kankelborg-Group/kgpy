import dataclasses
import typing as typ
from kgpy.optics.system.surface import coordinate
from ... import Child
from ..surface import SurfaceT
from . import Tilt, Decenter, TiltFirst

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class Base:

    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter())
    tilt_first: TiltFirst = dataclasses.field(default_factory=lambda: TiltFirst())


@dataclasses.dataclass
class TiltDecenter(Child[SurfaceT], coordinate.TiltDecenter, Base, typ.Generic[SurfaceT], ):

    def _update(self) -> typ.NoReturn:
        self.tilt = self.tilt
        self.decenter = self.decenter
        self.tilt_first = self.tilt_first

    @property
    def tilt(self) -> Tilt['TiltDecenter[SurfaceT]']:
        return self._tilt

    @tilt.setter
    def tilt(self, value: Tilt['TiltDecenter[SurfaceT]']):
        self._link(value)
        self._tilt = value

    @property
    def decenter(self) -> Decenter['TiltDecenter[SurfaceT]']:
        return self._decenter

    @decenter.setter
    def decenter(self, value: Decenter['TiltDecenter[SurfaceT]']):
        self._link(value)
        self._decenter = value

    @property
    def tilt_first(self) -> TiltFirst['TiltDecenter[SurfaceT]']:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: TiltFirst['TiltDecenter[SurfaceT]']):
        self._link(value)
        self._tilt_first = value

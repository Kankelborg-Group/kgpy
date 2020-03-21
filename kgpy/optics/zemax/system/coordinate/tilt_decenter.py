import dataclasses
import typing as typ
from kgpy.optics.system import coordinate
from ..descendants import Child
from .. import surface
from . import Tilt, Decenter, TiltFirst

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class Base:

    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter())
    tilt_first: TiltFirst = dataclasses.field(default_factory=lambda: TiltFirst())


@dataclasses.dataclass
class TiltDecenter(Child[surface.SurfaceT], coordinate.TiltDecenter, Base, typ.Generic[surface.SurfaceT], ):

    def _update(self) -> typ.NoReturn:
        self.tilt = self.tilt
        self.decenter = self.decenter
        self.tilt_first = self.tilt_first

    @property
    def tilt(self) -> Tilt:
        return self._tilt

    @tilt.setter
    def tilt(self, value: Tilt):
        value.parent = self
        self._tilt = value

    @property
    def decenter(self) -> Decenter:
        return self._decenter

    @decenter.setter
    def decenter(self, value: Decenter):
        value.parent = self
        self._decenter = value

    @property
    def tilt_first(self) -> TiltFirst:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: TiltFirst):
        value.parent = self
        self._tilt_first = value

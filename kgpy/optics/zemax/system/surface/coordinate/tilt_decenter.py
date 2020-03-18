import dataclasses
import typing as typ
from kgpy.optics.system import coordinate
from ... import surface
from . import Tilt, Decenter, TiltFirst

__all__ = ['TiltDecenter']

SurfaceT = typ.TypeVar('SurfaceT', bound=surface.Surface)


@dataclasses.dataclass
class Base(typ.Generic[SurfaceT]):
    surface: typ.Optional[SurfaceT] = None


class TiltDecenter(typ.Generic[ParentSurfaceT], Base, coordinate.TiltDecenter):

    def _update(self) -> None:
        self.tilt = self.tilt
        self.decenter = self.decenter
        self.tilt_first = self.tilt_first

    @property
    def tilt(self) -> Tilt:
        return self._tilt

    @tilt.setter
    def tilt(self, value: Tilt):
        value.tilt_decenter = self
        self._tilt = value

    @property
    def decenter(self) -> Decenter:
        return self._decenter

    @decenter.setter
    def decenter(self, value: Decenter):
        value.tilt_decenter = self
        self._decenter = value

    @property
    def tilt_first(self) -> TiltFirst:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: TiltFirst):
        value.tilt_decenter = self
        self._tilt_first = value

    @property
    def parent(self) -> SurfaceT:
        return self._surface

    @parent.setter
    def parent(self, value: SurfaceT):
        self._surface = value
        self._update()

import dataclasses
import typing as typ
from kgpy.optics.system import coordinate
from ... import Standard
from . import Tilt, Decenter

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class Base(coordinate.TiltDecenter):
    
    surface: typ.Optional[Standard] = None
    
    
class TiltDecenter(Base):
    
    def _update(self) -> None:
        self.tilt = self.tilt
        self.decenter = self.decenter

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

    #todo: put order here

    @property
    def surface(self) -> Standard:
        return self._surface

    @surface.setter
    def surface(self, value: Standard):
        self._surface = value
        self._update()

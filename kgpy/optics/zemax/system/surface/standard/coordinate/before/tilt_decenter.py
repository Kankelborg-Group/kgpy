import dataclasses
import typing as typ
from kgpy.optics.system import coordinate
from ... import Standard

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class Base(coordinate.TiltDecenter):
    
    surface: typ.Optional[Standard] = None
    
    
class TiltDecenter(Base):
    
    def _update(self) -> None:
        self.tilt = self.tilt
        self.decenter = self.decenter

    @property
    def surface(self) -> Standard:
        return self._surface

    @surface.setter
    def surface(self, value: Standard):
        self._surface = value
        self._update()

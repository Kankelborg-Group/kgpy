import dataclasses
import typing as typ

from kgpy.optics.system import coordinate
from ..surface import Surface
from . import Translate

__all__ = ['Transform']


@dataclasses.dataclass
class Base(coordinate.Transform):

    surface: typ.Optional[Surface] = None


class Transform(Base):
    
    def _update(self) -> None:
        self.translate = self.translate
    
    @property
    def translate(self) -> Translate:
        return self._translate
    
    @translate.setter
    def translate(self, value: Translate):
        self._translate = value
        value.transform = self
        
    @property
    def surface(self) -> Surface:
        return self._surface
    
    @surface.setter
    def surface(self, value: Surface):
        self._surface = value
        self._update()

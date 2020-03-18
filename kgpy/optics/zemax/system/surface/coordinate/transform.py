import dataclasses
import typing as typ

from kgpy.optics.system import coordinate
from .. import surface
from . import translate

__all__ = ['Transform']


@dataclasses.dataclass
class Base(coordinate.Transform):

    surface: 'typ.Optional[surface.Surface]' = None


class Transform(Base):
    
    def _update(self) -> None:
        self.translate = self.translate
    
    @property
    def translate(self) -> 'translate.Translate':
        return self._translate
    
    @translate.setter
    def translate(self, value: coordinate.Translate):
        value.transform = self
        self._translate = value

    @property
    def surface(self) -> 'surface.Surface':
        return self._surface
    
    @surface.setter
    def surface(self, value: 'surface.Surface'):
        self._surface = value
        self._update()

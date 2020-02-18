import dataclasses
import typing as typ

from kgpy.optics.system import name

from .surface import Surface

__all__ = ['Name']


@dataclasses.dataclass
class Base(name.Name):

    surface: Surface = None


class Name(Base):

    @classmethod
    def from_parent_instance(cls, parent: name.Name):
        self = cls()
        self.parent = parent.parent
        self.base = parent.base

    def _update(self) -> None:
        try:
            self.surface.lde_row.Comment = self.__str__()
        except AttributeError:
            pass

    @property
    def parent(self) -> 'typ.Optional[Name]':
        return self._parent

    @parent.setter
    def parent(self, value: 'typ.Optional[Name]'):
        self._parent = value
        self._update()

    @property
    def base(self) -> str:
        return self._base

    @base.setter
    def base(self, value: str):
        self._base = value
        self._update()

    @property
    def surface(self) -> Surface:
        return self._surface

    @surface.setter
    def surface(self, value: Surface):
        self._surface = value
        self._update()

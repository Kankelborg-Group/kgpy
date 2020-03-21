import abc
import dataclasses
import typing as typ
from . import surface

__all__ = ['Child', 'ParentT', 'SurfaceChildT']

ParentT = typ.TypeVar('ParentT')


@dataclasses.dataclass
class Base(typ.Generic[ParentT]):

    parent: typ.Optional[ParentT] = None


@dataclasses.dataclass
class Child(Base[ParentT], typ.Generic[ParentT], abc.ABC):

    @abc.abstractmethod
    def _update(self) -> typ.NoReturn:
        pass

    @property
    def parent(self) -> ParentT:
        return self._parent

    @parent.setter
    def parent(self, value: ParentT):
        self._parent = value
        self._update()


SurfaceChildT = typ.TypeVar('SurfaceChildT', bound='Child[surface.Surface]')

import abc
import dataclasses
import typing as typ
from . import surface   # need for bound on SurfaceT

__all__ = ['SurfaceT', 'Child', 'ChildT', 'Grandchild', 'GrandchildT']

ParentT = typ.TypeVar('ParentT')


@dataclasses.dataclass
class Base(typ.Generic[ParentT]):

    parent: typ.Optional[ParentT] = None


@dataclasses.dataclass
class Descendant(typ.Generic[ParentT], Base[ParentT]):

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


SurfaceT = typ.TypeVar('SurfaceT', bound='surface.Surface')


@dataclasses.dataclass
class Child(typ.Generic[SurfaceT], Descendant[SurfaceT]):

    pass


ChildT = typ.TypeVar('ChildT', bound=Child)

@dataclasses.dataclass
class Grandchild(typ.Generic[ChildT], Descendant[ChildT]):

    pass


GrandchildT = typ.TypeVar('GrandchildT', bound=Grandchild)

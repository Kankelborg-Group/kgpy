import abc
import dataclasses
import typing as typ

__all__ = ['ChildMixin']

BaseParentT = typ.TypeVar('BaseParentT')
ParentT = typ.TypeVar('ParentT')


@dataclasses.dataclass
class Base(typ.Generic[ParentT]):

    parent: typ.Optional[ParentT] = None


class ChildMixin(typ.Generic[ParentT], Base[ParentT]):

    @abc.abstractmethod
    def _update(self):
        pass

    @property
    def parent(self) -> ParentT:
        return self._parent

    @parent.setter
    def parent(self, value: ParentT):
        self._parent = value
        self._update()

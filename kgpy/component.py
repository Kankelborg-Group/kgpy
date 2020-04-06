import abc
import dataclasses
import typing as typ

__all__ = ['Component']

CompositeT = typ.TypeVar('CompositeT')


@dataclasses.dataclass
class Base(typ.Generic[CompositeT]):

    _composite: typ.Optional[CompositeT] = dataclasses.field(default=None, init=False, repr=False)


@dataclasses.dataclass
class Component(Base[CompositeT], typ.Generic[CompositeT], abc.ABC):

    @abc.abstractmethod
    def _update(self) -> typ.NoReturn:
        pass

    @property
    def _composite(self) -> CompositeT:
        return self.__composite

    @_composite.setter
    def _composite(self, value: CompositeT):
        self.__composite = value
        self._update()

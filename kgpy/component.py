import abc
import dataclasses
import typing as typ

__all__ = ['Component']

CompositeT = typ.TypeVar('CompositeT')


@dataclasses.dataclass
class Base(typ.Generic[CompositeT]):

    composite: typ.Optional[CompositeT] = None


@dataclasses.dataclass
class Component(Base[CompositeT], typ.Generic[CompositeT], abc.ABC):

    @abc.abstractmethod
    def _update(self) -> typ.NoReturn:
        pass

    @property
    def composite(self) -> CompositeT:
        return self._composite

    @composite.setter
    def composite(self, value: CompositeT):
        self._composite = value
        self._update()

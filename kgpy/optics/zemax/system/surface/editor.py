import dataclasses
import typing as typ

from ... import ZOSAPI
from ..system import System
from . import Surface

__all__ = ['Editor']


@dataclasses.dataclass
class Base:

    _surfaces: typ.List
    system: System


class Editor(Base):

    def _update(self) -> None:
        pass

    @property
    def _surfaces(self) -> typ.Iterable:
        return self.__surfaces

    @_surfaces.setter
    def _surfaces(self, value):
        pass

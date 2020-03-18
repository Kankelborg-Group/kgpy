import dataclasses
import typing as typ

from ... import ZOSAPI
from .. import system
from . import surface

__all__ = ['Editor']


@dataclasses.dataclass
class Base:

    _surfaces: 'typ.Iterable[surface.Surface]' = dataclasses.field(default_factory=lambda: [])
    system: 'system.System' = None


class Editor(Base):

    def _update(self) -> None:
        pass

    @property
    def _surfaces(self) -> typ.Iterable:
        return self._surfaces

    @_surfaces.setter
    def _surfaces(self, value):
        pass



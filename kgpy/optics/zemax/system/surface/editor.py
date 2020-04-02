import dataclasses
import typing as typ
from kgpy.component import Component
from .. import system as system_
from . import surface

__all__ = ['Editor']


@dataclasses.dataclass
class Base:

    _surfaces: 'typ.Iterable[surface.Surface]' = dataclasses.field(default_factory=lambda: [])


class Editor(Component[system_.System], Base):

    def _update(self) -> None:
        pass

    @property
    def _surfaces(self) -> typ.Iterable:
        return self._surfaces

    @_surfaces.setter
    def _surfaces(self, value):
        self._surfaces = value
        # todo: Need to loop through and set every surface
        raise NotImplementedError

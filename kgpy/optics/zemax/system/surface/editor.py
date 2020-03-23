import dataclasses
import typing as typ
EditorT = typ.TypeVar('EditorT', bound='Editor')
from .. import Child, system as system_
from . import surface

__all__ = ['Editor']


@dataclasses.dataclass
class Base:

    _surfaces: 'typ.Iterable[surface.Surface]' = dataclasses.field(default_factory=lambda: [])


class Editor(Child[system_.System], Base):

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

    @property
    def system(self) -> system_.System:
        return self.parent

    @system.setter
    def system(self, value: system_.System):
        self.parent = value



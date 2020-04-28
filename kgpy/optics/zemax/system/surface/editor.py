import dataclasses
import typing as typ
from kgpy.component import Component
from ... import ZOSAPI
from .. import system as system_
from . import surface

__all__ = ['Editor']


@dataclasses.dataclass
class Base:

    _surfaces: 'typ.Iterable[surface.Surface]' = dataclasses.field(default_factory=lambda: [])


class Editor(Component[system_.System], Base):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self._surfaces = self._surfaces

    @property
    def _surfaces(self) -> typ.List[surface.Surface]:
        return self.__surfaces

    @_surfaces.setter
    def _surfaces(self, value: typ.List[surface.Surface]):
        for v in value:
            v._composite = self
        self.__surfaces = value
        try:
            while self._zemax_lde.NumberOfSurfaces != len(self._surfaces):
                if self._zemax_lde.NumberOfSurfaces < len(self._surfaces):
                    self._zemax_lde.AddSurface()
                else:
                    self._zemax_lde.RemoveSurfaceAt(self._zemax_lde.NumberOfSurfaces)
        except AttributeError:
            pass

    @property
    def _zemax_lde(self) -> ZOSAPI.Editors.LDE.ILensDataEditor:
        return self._composite._zemax_system.LDE

    def index(self, surf: surface.Surface) -> int:
        self._surfaces.index(surf)

    def __iter__(self):
        return self._surfaces.__iter__()

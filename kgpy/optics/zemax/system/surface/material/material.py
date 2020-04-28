import dataclasses
import typing as typ
from kgpy.component import Component
from kgpy.optics import system
from ... import surface

__all__ = ['Material', 'NoMaterial']


@dataclasses.dataclass
class Material(Component['surface.Standard'], system.surface.Material):
    string: typ.ClassVar[str] = ''

    def _update(self) -> typ.NoReturn:
        super()._update()
        try:
            self._composite._lde_row.Material = self.string
        except AttributeError:
            pass


@dataclasses.dataclass
class NoMaterial(system.surface.material.NoMaterial, Material):
    pass

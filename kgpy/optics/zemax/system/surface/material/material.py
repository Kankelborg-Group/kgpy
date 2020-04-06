import dataclasses
import typing as typ
from kgpy.component import Component
from kgpy.optics import system
from ... import surface

__all__ = ['Material', 'NoMaterial']


@dataclasses.dataclass
class InstanceVarBase:
    string: str = dataclasses.field(default='', init=False, repr=False)


@dataclasses.dataclass
class Mixin(Component[surface.Standard], system.surface.Material):

    def _update(self) -> typ.NoReturn:
        self.string = self.string

    @property
    def string(self) -> str:
        return self._string

    @string.setter
    def string(self, value: str):
        self._string = value
        try:
            self._composite.lde_row.Material = value
        except AttributeError:
            pass


@dataclasses.dataclass
class Material(Mixin, InstanceVarBase):
    pass


@dataclasses.dataclass
class NoMaterial(Material):
    pass

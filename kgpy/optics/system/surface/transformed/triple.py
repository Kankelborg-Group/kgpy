import dataclasses
import typing as typ

from ... import Name, mixin, coordinate
from .. import Surface, Standard, Substrate
from . import Single

__all__ = ['Triple']

TransformedSurfacesT = typ.TypeVar('TransformedSurfacesT')


@dataclasses.dataclass
class Base(mixin.Named, typ.Generic[TransformedSurfacesT]):
    
    _all_surfaces: Single[Single[Single[TransformedSurfacesT]]] = dataclasses.field(
        init=False, 
        repr=False,
        default_factory=lambda: Single(
            name=Name(base='transform_1'), 
            surfaces=Single(
                name=Name(base='transform_2'),
                surfaces=Single(
                    name=Name(base='transform_3')
                )
            )
        )
    )
    
    surfaces: TransformedSurfacesT = None
    transform_1: coordinate.Transform = dataclasses.field(default_factory=coordinate.Transform())
    transform_2: coordinate.Transform = dataclasses.field(default_factory=coordinate.Transform())
    transform_3: coordinate.Transform = dataclasses.field(default_factory=coordinate.Transform())
    is_last_surface: bool = False
    
    def __iter__(self):
        return self._all_surfaces.__iter__()


class Triple(Base[TransformedSurfacesT]):
    
    @property
    def name(self) -> Name:
        return self._all_surfaces.name
    
    @name.setter
    def name(self, value: Name):
        self._all_surfaces.name.parent = value
        self._all_surfaces.surfaces.name.parent = value
        self._all_surfaces.surfaces.surfaces.name.parent = value

    @property
    def surfaces(self):
        return self._all_surfaces.surfaces.surfaces.surfaces

    @surfaces.setter
    def surfaces(self, value: TransformedSurfacesT):
        self._all_surfaces.surfaces.surfaces.surfaces = value

    @property
    def transform_1(self) -> coordinate.Transform:
        return self._all_surfaces.transform

    @transform_1.setter
    def transform_1(self, value: coordinate.Transform):
        self._all_surfaces.transform = value

    @property
    def transform_2(self) -> coordinate.Transform:
        return self._all_surfaces.surfaces.transform

    @transform_2.setter
    def transform_2(self, value: coordinate.Transform):
        self._all_surfaces.surfaces.transform = value

    @property
    def transform_3(self) -> coordinate.Transform:
        return self._all_surfaces.surfaces.surfaces.transform

    @transform_3.setter
    def transform_3(self, value: coordinate.Transform):
        self._all_surfaces.surfaces.surfaces.transform = value

    @property
    def is_last_surface(self) -> bool:
        return self._all_surfaces.is_last_surface

    @is_last_surface.setter
    def is_last_surface(self, value: bool):
        self._all_surfaces.is_last_surface = value
        self._all_surfaces.surfaces.is_last_surface = value
        self._all_surfaces.surfaces.surfaces.is_last_surface = value



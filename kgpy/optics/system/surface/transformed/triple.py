import dataclasses
import typing as typ

from ... import Name, mixin, coordinate
from .. import Surface, Standard, Substrate
from . import Single

__all__ = ['Triple']

TransformedSurfacesT = typ.TypeVar('TransformedSurfacesT')


@dataclasses.dataclass
class Base(mixin.Named, typ.Generic[TransformedSurfacesT]):
    
    _all_surfaces: Single[Single[Single[TransformedSurfacesT]]] = None


class Triple:

    

    @classmethod
    def from_properties(
            cls,
            name: Name,
            surfaces: TransformedSurfacesT,
            transform_1: typ.Optional[coordinate.Transform] = None,
            transform_2: typ.Optional[coordinate.Transform] = None,
            transform_err: typ.Optional[coordinate.Transform] = None,
            is_last_surface: bool = False,
    ) -> 'Triple[TransformedSurfacesT]':

        if transform_1 is None:
            transform_1 = coordinate.Transform()
        if transform_2 is None:
            transform_2 = coordinate.Transform()
        if transform_err is None:
            transform_err = coordinate.Transform()

        all_surfaces = Single(name=Name(name, '.transform_3'), surfaces=surfaces)
        all_surfaces = Single(name=Name(name, '.transform_2'), surfaces=all_surfaces)
        all_surfaces = Single(name=Name(name, '.transform_1'), surfaces=all_surfaces)

        self = cls(name=name, _all_surfaces=all_surfaces)

        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.transform_3 = transform_err
        self.is_last_surface = is_last_surface

        return self

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

    def __iter__(self):
        return self._all_surfaces.__iter__()

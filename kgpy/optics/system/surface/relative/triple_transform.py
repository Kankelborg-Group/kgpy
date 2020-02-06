import dataclasses
import typing as typ

from ... import Name, mixin, coordinate
from .. import Surface, Standard, Substrate
from . import SingleTransform

__all__ = ['TripleTransform']

TransformedSurfacesT = typ.TypeVar('TransformedSurfacesT')


@dataclasses.dataclass
class TripleTransform(mixin.Named, typ.Generic[TransformedSurfacesT]):

    _transforms: SingleTransform[SingleTransform[SingleTransform[TransformedSurfacesT]]] = dataclasses.field(
        default_factory=SingleTransform(surfaces=SingleTransform(surfaces=SingleTransform(surfaces=Standard()))))

    @classmethod
    def from_properties(
            cls,
            name: Name,
            surfaces: TransformedSurfacesT,
            transform_1: typ.Optional[coordinate.Transform] = None,
            transform_2: typ.Optional[coordinate.Transform] = None,
            transform_err: typ.Optional[coordinate.Transform] = None,
            is_last_surface: bool = False,
    ) -> 'TripleTransform[TransformedSurfacesT]':

        if transform_1 is None:
            transform_1 = coordinate.Transform()
        if transform_2 is None:
            transform_2 = coordinate.Transform()
        if transform_err is None:
            transform_err = coordinate.Transform()

        a = surfaces
        b = SingleTransform(name=Name(name, '.transform_3'), surfaces=a)
        c = SingleTransform(name=Name(name, '.transform_2'), surfaces=b)
        d = SingleTransform(name=Name(name, '.transform_1'), surfaces=c)

        self = cls(name=name)

        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.transform_3 = transform_err
        self.is_last_surface = is_last_surface

        return self

    @property
    def surfaces(self):
        return self._transforms.surfaces.surfaces.surfaces

    @surfaces.setter
    def surfaces(self, value: TransformedSurfacesT):
        self._transforms.surfaces.surfaces.surfaces = value

    @property
    def transform_1(self) -> coordinate.Transform:
        return self._transforms.transform

    @transform_1.setter
    def transform_1(self, value: coordinate.Transform):
        self._transforms.transform = value

    @property
    def transform_2(self) -> coordinate.Transform:
        return self._transforms.surfaces.transform

    @transform_2.setter
    def transform_2(self, value: coordinate.Transform):
        self._transforms.surfaces.transform = value

    @property
    def transform_3(self) -> coordinate.Transform:
        return self._transforms.surfaces.surfaces.transform

    @transform_3.setter
    def transform_3(self, value: coordinate.Transform):
        self._transforms.surfaces.surfaces.transform = value

    @property
    def is_last_surface(self) -> bool:
        return self._transforms.is_last_surface

    @is_last_surface.setter
    def is_last_surface(self, value: bool):
        self._transforms.is_last_surface = value
        self._transforms.surfaces.is_last_surface = value
        self._transforms.surfaces.surfaces.is_last_surface = value

    def __iter__(self):
        return self._transforms.__iter__()

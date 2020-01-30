import dataclasses
import typing as typ
import numpy as np

from kgpy.optics import system

from . import relative

__all__ = ['Surface']

ApertureSurfaceT = typ.TypeVar('ApertureSurfaceT', bound=system.surface.Standard)
MainSurfaceT = typ.TypeVar('MainSurfaceT', bound=system.surface.Standard)

ClearAperT = typ.TypeVar('ClearAperT', bound=system.surface.Aperture)


@dataclasses.dataclass
class Mechanical(typ.Generic[ApertureSurfaceT, MainSurfaceT]):
    """
    Representation of a real optical surface.
    This class uses two `system.Surface` objects to model a more realistic optical surface.
    The first `system.Surface` object, `self.main_surface` represents the clear aperture.
    The second `system.Surface` object, `self.substrate_surface` represents the mechanical aperture and the substrate.
    """

    aperture_surface: ApertureSurfaceT
    main_surface: MainSurfaceT

    def __iter__(self) -> typ.Generator[system.Surface]:
        yield self.aperture_surface
        yield self.main_surface


@dataclasses.dataclass
class Surface(system.mixin.Named, typ.Generic[ApertureSurfaceT, MainSurfaceT]):
    """
    A `Mechanical`
    """

    _surfaces: relative.GenericSurfaces[relative.GenericSurfaces[relative.GenericSurfaces[Mechanical[ApertureSurfaceT,
                                                                                                     MainSurfaceT]]]]

    @classmethod
    def from_properties(
            cls,
            name: str,
            aper_surface: ApertureSurfaceT,
            main_surface: MainSurfaceT,
            transform_1: typ.Optional[system.coordinate.Transform] = None,
            transform_2: typ.Optional[system.coordinate.Transform] = None,
            transform_err: typ.Optional[system.coordinate.Transform] = None,
            is_last_surface: bool = False,
    ) -> 'Surface[ApertureSurfaceT, MainSurfaceT]':

        if transform_1 is None:
            transform_1 = system.coordinate.Transform()
        if transform_2 is None:
            transform_2 = system.coordinate.Transform()
        if transform_err is None:
            transform_err = system.coordinate.Transform()

        a = Mechanical(aperture_surface=aper_surface, main_surface=main_surface)
        b = relative.GenericSurfaces(name=name + '.relative_err', main=a)
        c = relative.GenericSurfaces(name=name + '.relative_2', main=b)
        d = relative.GenericSurfaces(name=name + '.relative_1', main=c)

        self = cls(name=name, _surfaces=d)

        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.transform_err = transform_err
        self.is_last_surface = is_last_surface

        return self

    @property
    def main_surface(self) -> MainSurfaceT:
        return self._surfaces.main.main.main.main_surface

    @property
    def aperture_surface(self) -> ApertureSurfaceT:
        return self._surfaces.main.main.main.aperture_surface

    @property
    def transform_1(self) -> system.coordinate.Transform:
        return self._surfaces.transform

    @transform_1.setter
    def transform_1(self, value: system.coordinate.Transform):
        self._surfaces.transform = value

    @property
    def transform_2(self) -> system.coordinate.Transform:
        return self._surfaces.main.transform

    @transform_2.setter
    def transform_2(self, value: system.coordinate.Transform):
        self._surfaces.main.transform = value

    @property
    def transform_err(self) -> system.coordinate.Transform:
        return self._surfaces.main.main.transform

    @transform_err.setter
    def transform_err(self, value: system.coordinate.Transform):
        self._surfaces.main.main.transform = value

    @property
    def is_last_surface(self) -> bool:
        return self._surfaces.is_last_surface

    @is_last_surface.setter
    def is_last_surface(self, value: bool):
        self._surfaces.is_last_surface = value
        self._surfaces.main.is_last_surface = value
        self._surfaces.main.main.is_last_surface = value

    def __iter__(self):
        return self._surfaces.__iter__()



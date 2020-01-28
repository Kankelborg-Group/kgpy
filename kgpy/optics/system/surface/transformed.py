import dataclasses
import typing as typ
import numpy as np

from kgpy.optics import system

from .relative import GenericSurfaces as RSurfs

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

    _surfaces: RSurfs[RSurfs[RSurfs[Mechanical[ApertureSurfaceT, MainSurfaceT]]]]

    @classmethod
    def from_properties(
            cls,
            name: np.ndarray[str],
            aper_surface: ApertureSurfaceT,
            main_surface: MainSurfaceT,
            transform_1: typ.Optional[system.coordinate.Transform] = None,
            transform_2: typ.Optional[system.coordinate.Transform] = None,
            transform_err: typ.Optional[system.coordinate.Transform] = None,
    ) -> 'Surface[ApertureSurfaceT, MainSurfaceT]':

        if transform_1 is None:
            transform_1 = system.coordinate.Transform()
        if transform_2 is None:
            transform_2 = system.coordinate.Transform()
        if transform_err is None:
            transform_err = system.coordinate.Transform()

        a = Mechanical(aperture_surface=aper_surface, main_surface=main_surface)
        b = RSurfs.from_cbreak_args(name=name + '.relative_err', main=a, transform=transform_err,)
        c = RSurfs.from_cbreak_args(name=name + '.relative_2', main=b, transform=transform_2,)
        d = RSurfs.from_cbreak_args(name=name + '.relative_1', main=c, transform=transform_1,)

        return cls(name=name, _surfaces=d)

    @classmethod
    def from_clear_aper(
            cls,
            name: np.ndarray[str],
            clear_aperture: ClearAperT,
            main_surface: MainSurfaceT,
            transform_1: typ.Optional[system.coordinate.Transform] = None,
            transform_2: typ.Optional[system.coordinate.Transform] = None,
            transform_err: typ.Optional[system.coordinate.Transform] = None,
    ):
        aper_surface = dataclasses.replace(
            main_surface,
            name=name+'.aperture',
            aperture=clear_aperture,
            material=system.surface.material.NoMaterial(),
        )
        return cls.from_properties(name, aper_surface, main_surface, transform_1, transform_2, transform_err)

    @property
    def main_surface(self) -> MainSurfaceT:
        return self._surfaces.main.main.main.main_surface

    @main_surface.setter
    def main_surface(self, value: MainSurfaceT):
        self._surfaces.main.main.main.main_surface = value

    @property
    def aperture_surface(self) -> ApertureSurfaceT:
        return self._surfaces.main.main.main.aperture_surface

    @aperture_surface.setter
    def aperture_surface(self, value: ApertureSurfaceT):
        self._surfaces.main.main.main.aperture_surface = value

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

    def __iter__(self):
        return self._surfaces.__iter__()



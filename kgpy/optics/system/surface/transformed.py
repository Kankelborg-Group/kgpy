import dataclasses
import typing as typ

from kgpy.optics.system.name import Name

from .. import Name, mixin, coordinate
from . import Surface, Standard, Aperture

from . import relative

__all__ = ['Surfaces']

ApertureSurfaceT = typ.TypeVar('ApertureSurfaceT', bound=Standard)
MainSurfaceT = typ.TypeVar('MainSurfaceT', bound=Standard)

ClearAperT = typ.TypeVar('ClearAperT', bound=Aperture)


@dataclasses.dataclass
class Mechanical(typ.Generic[ApertureSurfaceT, MainSurfaceT]):
    """
    Representation of a real optical surface.
    This class uses two `system.Surface` objects to model a more realistic optical surface.
    The first `system.Surface` object, `self.main_surface` represents the clear aperture.
    The second `system.Surface` object, `self.substrate_surface` represents the mechanical aperture and the substrate.
    """

    aperture_surface: ApertureSurfaceT = dataclasses.field(default_factory=lambda: Standard())
    main_surface: MainSurfaceT = dataclasses.field(default_factory=lambda: Standard())

    def __iter__(self) -> typ.Iterator[Surface]:
        yield from self.aperture_surface
        yield from self.main_surface


@dataclasses.dataclass
class Surfaces(mixin.Named, typ.Generic[ApertureSurfaceT, MainSurfaceT]):
    """
    A `Mechanical`
    """

    _surfaces: relative.GenericSurfaces[
        relative.GenericSurfaces[
            relative.GenericSurfaces[
                Mechanical[ApertureSurfaceT, MainSurfaceT]
            ]
        ]
    ] = dataclasses.field(default_factory=relative.GenericSurfaces(
        main=relative.GenericSurfaces(
            main=relative.GenericSurfaces(
                main=Mechanical()
            )
        )
    ))

    @classmethod
    def from_properties(
            cls,
            name: Name,
            aper_surface: ApertureSurfaceT,
            main_surface: MainSurfaceT,
            transform_1: typ.Optional[coordinate.Transform] = None,
            transform_2: typ.Optional[coordinate.Transform] = None,
            transform_err: typ.Optional[coordinate.Transform] = None,
            is_last_surface: bool = False,
    ) -> 'Surfaces[ApertureSurfaceT, MainSurfaceT]':

        if transform_1 is None:
            transform_1 = coordinate.Transform()
        if transform_2 is None:
            transform_2 = coordinate.Transform()
        if transform_err is None:
            transform_err = coordinate.Transform()

        a = Mechanical(aperture_surface=aper_surface, main_surface=main_surface)
        b = relative.GenericSurfaces(name=Name(name, '.relative_err'), main=a)
        c = relative.GenericSurfaces(name=Name(name, '.relative_2'), main=b)
        d = relative.GenericSurfaces(name=Name(name, '.relative_1'), main=c)

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
    def transform_1(self) -> coordinate.Transform:
        return self._surfaces.transform

    @transform_1.setter
    def transform_1(self, value: coordinate.Transform):
        self._surfaces.transform = value

    @property
    def transform_2(self) -> coordinate.Transform:
        return self._surfaces.main.transform

    @transform_2.setter
    def transform_2(self, value: coordinate.Transform):
        self._surfaces.main.transform = value

    @property
    def transform_err(self) -> coordinate.Transform:
        return self._surfaces.main.main.transform

    @transform_err.setter
    def transform_err(self, value: coordinate.Transform):
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

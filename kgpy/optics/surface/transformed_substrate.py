import typing as typ
import dataclasses
from . import Standard, Substrate, Transformed

__all__ = ['TransformedSubstrate']

ApertureSurfaceT = typ.TypeVar('ApertureSurfaceT', bound=Standard)
MainSurfaceT = typ.TypeVar('MainSurfaceT', bound=Standard)


@dataclasses.dataclass
class TransformedSubstrate(typ.Generic[ApertureSurfaceT, MainSurfaceT]):

    _surfaces: Transformed[Substrate[ApertureSurfaceT, MainSurfaceT]]

    @property
    def _aper(self) -> ApertureSurfaceT:
        return self._surfaces.surfaces.aperture_surface

    @property
    def _main(self) -> MainSurfaceT:
        return self._surfaces.surfaces.main_surface

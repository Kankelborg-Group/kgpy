import dataclasses
import typing as typ
from . import Standard
from kgpy.optics import system

__all__ = ['Substrate']

ApertureSurfaceT = typ.TypeVar('ApertureSurfaceT', bound=Standard)
MainSurfaceT = typ.TypeVar('MainSurfaceT', bound=Standard)


@dataclasses.dataclass
class Substrate(system.surface.Substrate[ApertureSurfaceT, MainSurfaceT]):

    pass

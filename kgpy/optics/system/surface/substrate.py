import dataclasses
import typing as typ
from .. import mixin
from . import Surface, Standard

ApertureSurfaceT = typ.TypeVar('ApertureSurfaceT', bound=Standard)
MainSurfaceT = typ.TypeVar('MainSurfaceT', bound=Standard)


@dataclasses.dataclass
class Substrate(mixin.ZemaxCompatible, typ.Generic[ApertureSurfaceT, MainSurfaceT]):
    """
    Representation of a real optical surface.
    This class uses two `system.Surface` objects to model a more realistic optical surface.
    The first `system.Surface` object, `self.main_surface` represents the clear aperture.
    The second `system.Surface` object, `self.substrate_surface` represents the mechanical aperture and the substrate.
    """

    aperture_surface: ApertureSurfaceT = dataclasses.field(default_factory=lambda: Standard())
    main_surface: MainSurfaceT = dataclasses.field(default_factory=lambda: Standard())

    def to_zemax(self) -> 'Substrate[ApertureSurfaceT, MainSurfaceT]':
        from kgpy.optics import zemax
        return zemax.system.surface.Substrate(
            aperture_surface=self.aperture_surface.to_zemax(),
            main_surface=self.main_surface.to_zemax(),
        )

    def __iter__(self) -> typ.Iterator[Surface]:
        yield from self.aperture_surface
        yield from self.main_surface

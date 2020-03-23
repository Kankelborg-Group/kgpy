import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from . import coordinate, Surface, material as material_, aperture as aperture_

__all__ = ['Standard']

MaterialType = typ.TypeVar('MaterialType', bound=material_.Material)
ApertureType = typ.TypeVar('ApertureType', bound=aperture_.Aperture)


@dataclasses.dataclass
class Standard(Surface, typ.Generic[MaterialType, ApertureType]):

    radius: u.Quantity = np.inf * u.mm
    conic: u.Quantity = 0 * u.dimensionless_unscaled
    material: MaterialType = dataclasses.field(default_factory=lambda: material_.NoMaterial())
    aperture: ApertureType = dataclasses.field(default_factory=lambda: aperture_.NoAperture())
    transform_before: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter())
    transform_after: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter())

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius,
            self.conic,
            self.material.config_broadcast,
            self.aperture.config_broadcast,
            self.transform_before.config_broadcast,
            self.transform_after.config_broadcast,
        )

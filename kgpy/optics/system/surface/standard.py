import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

from .. import coordinate
from . import Surface, material as material_module, aperture as aperture_module

__all__ = ['Standard']

MaterialType = typ.TypeVar('MaterialType', bound=material_module.Material)
ApertureType = typ.TypeVar('ApertureType', bound=aperture_module.Aperture)


@dataclasses.dataclass
class Standard(Surface, typ.Generic[MaterialType, ApertureType]):

    radius: u.Quantity = np.inf * u.mm
    conic: u.Quantity = 0 * u.dimensionless_unscaled
    material: MaterialType = dataclasses.field(default_factory=lambda: material_module.NoMaterial())
    aperture: ApertureType = dataclasses.field(default_factory=lambda: aperture_module.NoAperture())
    transform_before: coordinate.Transform = dataclasses.field(default_factory=lambda: coordinate.Transform())
    transform_after: coordinate.Transform = dataclasses.field(default_factory=lambda: coordinate.Transform())

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius,
            self.conic,
            self.material.config_broadcast(),
            self.aperture.config_broadcast(),
            self.transform_before.config_broadcast(),
            self.transform_after.config_broadcast(),
        )

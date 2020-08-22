import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import kgpy.vector
from kgpy.vector import x, y, z
from .. import Rays, material, aperture
from . import Standard

__all__ = ['DiffractionGrating']

MaterialT = typ.TypeVar('MaterialT', bound=material.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture.Aperture)


@dataclasses.dataclass
class DiffractionGrating(Standard[MaterialT, ApertureT]):

    diffraction_order: u.Quantity = 1 * u.dimensionless_unscaled
    groove_density: u.Quantity = 0 * (1 / u.mm)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.diffraction_order,
            self.groove_density,
        )

    def groove_normal(self, sx: u.Quantity, sy: u.Quantity) -> u.Quantity:
        return u.Quantity([0 << 1 / u.mm, self.groove_density, 0 << 1 / u.mm])

    def _calc_input_vector(self, rays: Rays) -> u.Quantity:
        n1 = rays.index_of_refraction
        n2 = self._index_of_refraction(rays)
        a = n1 * rays.direction / n2
        normal = self.groove_normal(rays.position[x], rays.position[y])
        return a + self.diffraction_order * rays.wavelength * normal

    def _calc_input_direction(self, rays: Rays) -> u.Quantity:
        return kgpy.vector.normalize(self._calc_input_vector(rays))

    def _calc_index_ratio(self, rays: Rays) -> u.Quantity:
        return kgpy.vector.length(self._calc_input_vector(rays))

    def wavelength_from_angles(
            self,
            input_angle: u.Quantity,
            output_angle: u.Quantity,
    ) -> u.Quantity:
        a = np.sin(input_angle) + np.sin(output_angle)
        return a / (self.diffraction_order * self.groove_density)

    def diffraction_angle(
            self,
            wavelength: u.Quantity,
            input_angle: u.Quantity,
    ) -> u.Quantity:
        return np.arcsin(self.diffraction_order * wavelength * self.groove_density - np.sin(input_angle)) << u.rad

    def copy(self) -> 'DiffractionGrating':
        return DiffractionGrating(
            name=self.name.copy(),
            thickness=self.thickness.copy(),
            is_active=self.is_active.copy(),
            is_visible=self.is_visible.copy(),
            radius=self.radius.copy(),
            conic=self.conic.copy(),
            material=self.material.copy(),
            aperture=self.aperture.copy(),
            transform_before=self.transform_before.copy(),
            transform_after=self.transform_after.copy(),
            intercept_error=self.intercept_error.copy(),
            diffraction_order=self.diffraction_order.copy(),
            groove_density=self.groove_density.copy(),
        )

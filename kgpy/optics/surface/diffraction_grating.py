import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from kgpy import vector, optics
from kgpy.vector import x, y, z
from . import Standard

__all__ = ['DiffractionGrating']

MaterialT = typ.TypeVar('MaterialT', bound=optics.Material)
ApertureT = typ.TypeVar('ApertureT', bound=optics.Aperture)
ApertureMechT = typ.TypeVar('ApertureMechT', bound=optics.Aperture)


@dataclasses.dataclass
class DiffractionGrating(Standard[MaterialT, ApertureT, ApertureMechT]):

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
        return vector.from_components(y=self.groove_density)

    def _calc_input_vector(self, rays: optics.Rays) -> u.Quantity:
        n1 = rays.index_of_refraction
        n2 = self._index_of_refraction(rays)
        a = n1 * rays.direction / n2
        normal = self.groove_normal(rays.position[x], rays.position[y])
        return a + self.diffraction_order * rays.wavelength * normal

    def _calc_input_direction(self, rays: optics.Rays) -> u.Quantity:
        return vector.normalize(self._calc_input_vector(rays))

    def _calc_index_ratio(self, rays: optics.Rays) -> u.Quantity:
        return vector.length(self._calc_input_vector(rays))

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
        other = super().copy()      # type: DiffractionGrating
        other.diffraction_order = self.diffraction_order.copy()
        other.groove_density = self.groove_density.copy()
        return other

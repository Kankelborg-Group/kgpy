import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import vector
from .. import Rays
from ..material import Material
from . import Rulings

__all__ = ['ConstantDensity']


@dataclasses.dataclass
class ConstantDensity(Rulings):
    diffraction_order: u.Quantity = 1 * u.dimensionless_unscaled
    ruling_density: u.Quantity = 0 * (1 / u.mm)

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return vector.from_components(y=self.ruling_density)

    def _effective_input_vector(
            self,
            rays: Rays,
            material: typ.Optional[Material] = None,
    ) -> u.Quantity:
        if material is not None:
            n2 = material.index_of_refraction(rays)
        else:
            n2 = np.sign(rays.index_of_refraction) << u.dimensionless_unscaled
        a = rays.index_of_refraction * rays.direction
        normal = self.normal(rays.position[vector.x], rays.position[vector.y])
        return a + n2 * self.diffraction_order * rays.wavelength * normal

    def wavelength_from_angles(
            self,
            input_angle: u.Quantity,
            output_angle: u.Quantity,
    ) -> u.Quantity:
        a = np.sin(input_angle) + np.sin(output_angle)
        return a / (self.diffraction_order * self.ruling_density)

    def diffraction_angle(
            self,
            wavelength: u.Quantity,
            input_angle: u.Quantity,
    ) -> u.Quantity:
        return np.arcsin(self.diffraction_order * wavelength * self.ruling_density - np.sin(input_angle)) << u.rad

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.diffraction_order)
        out = np.broadcast(out, self.ruling_density)
        return out

    def copy(self) -> 'ConstantDensity':
        other = super().copy()  # type: ConstantDensity
        other.diffraction_order = self.diffraction_order.copy()
        other.ruling_density = self.ruling_density.copy()
        return other

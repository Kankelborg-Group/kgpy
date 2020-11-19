import typing as typ
import abc
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import mixin, vector
from ..rays import Rays
from .material import Material

__all__ = ['Rulings', 'ConstantDensity', 'CubicPolyDensity']


class Rulings(
    mixin.Broadcastable,
    mixin.Copyable,
    abc.ABC,
):

    @abc.abstractmethod
    def normal(self, x: u.Quantity, y: u.Quantity):
        pass

    @abc.abstractmethod
    def _effective_input_vector(
            self,
            rays: Rays,
            material: typ.Optional[Material] = None,
    ) -> u.Quantity:
        pass

    def effective_input_direction(
            self,
            rays: Rays,
            material: typ.Optional[Material] = None,
    ):
        return vector.normalize(self._effective_input_vector(rays, material=material))

    def effective_input_index(
            self,
            rays: Rays,
            material: typ.Optional[Material] = None,
    ):
        return vector.length(self._effective_input_vector(rays, material=material))


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


@dataclasses.dataclass
class CubicPolyDensity(ConstantDensity):
    ruling_density_linear: u.Quantity = 0 / (u.mm ** 2)
    ruling_density_quadratic: u.Quantity = 0 / (u.mm ** 3)
    ruling_density_cubic: u.Quantity = 0 / (u.mm ** 4)

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        x2 = np.square(x)
        term0 = self.ruling_density[..., None, None, None, None, None]
        term1 = self.ruling_density_linear * x
        term2 = self.ruling_density_quadratic * x2
        term3 = self.ruling_density_cubic * x * x2
        groove_density = term0 + term1 + term2 + term3
        return vector.from_components(x=groove_density)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.ruling_density_linear)
        out = np.broadcast(out, self.ruling_density_quadratic)
        out = np.broadcast(out, self.ruling_density_cubic)
        return out

    def copy(self) -> 'CubicPolyDensity':
        other = super().copy()  # type: CubicPolyDensity
        other.ruling_density_linear = self.ruling_density_linear.copy()
        other.ruling_density_quadratic = self.ruling_density_quadratic.copy()
        other.ruling_density_cubic = self.ruling_density_cubic.copy()
        return other


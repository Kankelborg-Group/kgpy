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
    def normal(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0, ):
        pass

    @abc.abstractmethod
    def effective_input_vector(
            self,
            rays: Rays,
            material: typ.Optional[Material] = None,
    ) -> vector.Vector3D:
        pass

    @classmethod
    def effective_input_direction(cls, input_vector: vector.Vector3D):
        return input_vector.normalize()

    @classmethod
    def effective_input_index(cls, input_vector: vector.Vector3D):
        return input_vector.length

    # def effective_input_direction(
    #         self,
    #         rays: Rays,
    #         material: typ.Optional[Material] = None,
    # ) -> vector.Vector3D:
    #     return self.effective_input_vector(rays=rays, material=material).normalize()
    #
    # def effective_input_index(
    #         self,
    #         rays: Rays,
    #         material: typ.Optional[Material] = None,
    # ) -> u.Quantity:
    #     return self.effective_input_vector(rays=rays, material=material).length


@dataclasses.dataclass
class ConstantDensity(Rulings):
    diffraction_order: u.Quantity = 1 * u.dimensionless_unscaled
    ruling_density: u.Quantity = 0 * (1 / u.mm)

    def __eq__(self, other: 'ConstantDensity') -> bool:
        if not super().__eq__(other):
            return False
        if not (other.diffraction_order == self.diffraction_order).all():
            return False
        if not (other.ruling_density == self.ruling_density).all():
            return False
        return True

    @property
    def ruling_spacing(self) -> u.Quantity:
        return 1 / self.ruling_density

    def normal(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0) -> vector.Vector3D:
        extra_dims_slice = (Ellipsis, ) + num_extra_dims * (np.newaxis, )
        return vector.Vector3D(
            x=0 * self.ruling_density,
            y=self.ruling_density[extra_dims_slice],
            z=0 * self.ruling_density
        )

    def effective_input_vector(
            self,
            rays: Rays,
            material: typ.Optional[Material] = None,
    ) -> vector.Vector3D:
        num_extra_dims = rays.axis.ndim
        extra_dims_slice = (Ellipsis,) + num_extra_dims * (np.newaxis,)
        if material is not None:
            n2 = material.index_of_refraction(rays)
        else:
            n2 = np.sign(rays.index_of_refraction) << u.dimensionless_unscaled
        a = rays.index_of_refraction * rays.direction
        normal = self.normal(rays.position.x, rays.position.y, num_extra_dims=num_extra_dims)
        return a + n2 * self.diffraction_order[extra_dims_slice] * rays.wavelength * normal

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

    def view(self) -> 'ConstantDensity':
        other = super().view()  # type: ConstantDensity
        other.diffraction_order = self.diffraction_order
        other.ruling_density = self.ruling_density
        return other

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

    def normal(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0, ) -> vector.Vector3D:
        extra_dims_slice = (Ellipsis,) + num_extra_dims * (np.newaxis,)
        x2 = np.square(x)
        term0 = self.ruling_density[extra_dims_slice]
        term1 = self.ruling_density_linear[extra_dims_slice] * x
        term2 = self.ruling_density_quadratic[extra_dims_slice] * x2
        term3 = self.ruling_density_cubic[extra_dims_slice] * x * x2
        groove_density = term0 + term1 + term2 + term3
        return vector.Vector3D(x=groove_density, y=0 * groove_density, z=0 * groove_density)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.ruling_density_linear)
        out = np.broadcast(out, self.ruling_density_quadratic)
        out = np.broadcast(out, self.ruling_density_cubic)
        return out

    def view(self) -> 'CubicPolyDensity':
        other = super().view()  # type: CubicPolyDensity
        other.ruling_density_linear = self.ruling_density_linear
        other.ruling_density_quadratic = self.ruling_density_quadratic
        other.ruling_density_cubic = self.ruling_density_cubic
        return other

    def copy(self) -> 'CubicPolyDensity':
        other = super().copy()  # type: CubicPolyDensity
        other.ruling_density_linear = self.ruling_density_linear.copy()
        other.ruling_density_quadratic = self.ruling_density_quadratic.copy()
        other.ruling_density_cubic = self.ruling_density_cubic.copy()
        return other


@dataclasses.dataclass
class CubicPolySpacing(ConstantDensity):
    ruling_spacing_linear: u.Quantity = 0 * u.dimensionless_unscaled
    ruling_spacing_quadratic: u.Quantity = 0 / (u.mm ** 1)
    ruling_spacing_cubic: u.Quantity = 0 / (u.mm ** 2)

    def normal(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0, ) -> vector.Vector3D:
        extra_dims_slice = (Ellipsis,) + num_extra_dims * (np.newaxis,)
        x2 = np.square(x)
        term0 = self.ruling_spacing[extra_dims_slice]
        term1 = self.ruling_spacing_linear[extra_dims_slice] * x
        term2 = self.ruling_spacing_quadratic[extra_dims_slice] * x2
        term3 = self.ruling_spacing_cubic[extra_dims_slice] * x * x2
        ruling_spacing = term0 + term1 + term2 + term3
        groove_density = 1 / ruling_spacing
        return vector.Vector3D(x=groove_density, y=0 * groove_density, z=0 * groove_density)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.ruling_spacing_linear)
        out = np.broadcast(out, self.ruling_spacing_quadratic)
        out = np.broadcast(out, self.ruling_spacing_cubic)
        return out

    def view(self) -> 'CubicPolySpacing':
        other = super().view()  # type: CubicPolyDensity
        other.ruling_spacing_linear = self.ruling_spacing_linear
        other.ruling_spacing_quadratic = self.ruling_spacing_quadratic
        other.ruling_spacing_cubic = self.ruling_spacing_cubic
        return other

    def copy(self) -> 'CubicPolySpacing':
        other = super().copy()  # type: CubicPolyDensity
        other.ruling_spacing_linear = self.ruling_spacing_linear.copy()
        other.ruling_spacing_quadratic = self.ruling_spacing_quadratic.copy()
        other.ruling_spacing_cubic = self.ruling_spacing_cubic.copy()
        return other

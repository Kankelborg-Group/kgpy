import typing as typ
import abc
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.mixin
import kgpy.uncertainty
import kgpy.vectors
from .. import rays
from . import materials

__all__ = [
    'Ruling',
    'ConstantDensity',
    'CubicPolyDensity',
    'CubicPolySpacing',
]

RulingT = typ.TypeVar('RulingT', bound='Ruling')
ConstantDensityT = typ.TypeVar('ConstantDensityT', bound='ConstantDensity')
CubicPolyDensityT = typ.TypeVar('CubicPolyDensityT', bound='CubicPolyDensity')
CubicPolySpacingT = typ.TypeVar('CubicPolySpacingT', bound='CubicPolySpacing')


class Ruling(
    kgpy.mixin.Broadcastable,
    kgpy.mixin.Copyable,
    abc.ABC,
):

    @abc.abstractmethod
    def normal(self: RulingT, position: kgpy.vectors.Cartesian2D) -> kgpy.vectors.Cartesian3D:
        pass

    @abc.abstractmethod
    def effective_input_vector(
            self: RulingT,
            ray: rays.RayVector,
            material: typ.Optional[materials.Material] = None,
    ) -> kgpy.vectors.Cartesian3D:
        pass

    @classmethod
    def effective_input_direction(
            cls: typ.Type[RulingT],
            input_vector: kgpy.vectors.Cartesian3D,
    ) -> kgpy.vectors.Cartesian3D:
        return input_vector.normalized

    @classmethod
    def effective_input_index(
            cls: typ.Type[RulingT],
            input_vector: kgpy.vectors.Cartesian3D,
    ) -> kgpy.uncertainty.ArrayLike:
        return input_vector.length


@dataclasses.dataclass
class ConstantDensity(Ruling):
    diffraction_order: u.Quantity = 1 * u.dimensionless_unscaled
    ruling_density: kgpy.uncertainty.ArrayLike = 0 * (1 / u.mm)

    def __eq__(self: ConstantDensityT, other: ConstantDensityT) -> bool:
        if not super().__eq__(other):
            return False
        if not np.all(other.diffraction_order == self.diffraction_order):
            return False
        if not np.all(other.ruling_density == self.ruling_density):
            return False
        return True

    @property
    def ruling_spacing(self: ConstantDensityT) -> kgpy.uncertainty.ArrayLike:
        return 1 / self.ruling_density

    def normal(self, position: kgpy.vectors.Cartesian3D) -> kgpy.vectors.Cartesian3D:
        return kgpy.vectors.Cartesian3D(
            x=0 * self.ruling_density,
            y=self.ruling_density,
            z=0 * self.ruling_density
        )

    def effective_input_vector(
            self: ConstantDensityT,
            ray: rays.RayVector,
            material: typ.Optional[materials.Material] = None,
    ) -> kgpy.vectors.Cartesian3D:
        if material is not None:
            n2 = material.index_refraction(ray)
        else:
            n2 = np.sign(ray.index_refraction) << u.dimensionless_unscaled
        a = ray.index_refraction * ray.direction
        normal = self.normal(ray.position.xy)
        return a + n2 * self.diffraction_order * ray.wavelength * normal

    def wavelength_from_angles(
            self: ConstantDensityT,
            input_angle: kgpy.uncertainty.ArrayLike,
            output_angle: kgpy.uncertainty.ArrayLike,
    ) -> kgpy.uncertainty.ArrayLike:
        a = np.sin(input_angle) + np.sin(output_angle)
        return a / (self.diffraction_order * self.ruling_density)

    def diffraction_angle(
            self: ConstantDensityT,
            wavelength: kgpy.uncertainty.ArrayLike,
            input_angle: kgpy.uncertainty.ArrayLike,
    ) -> kgpy.uncertainty.ArrayLike:
        return np.arcsin(self.diffraction_order * wavelength * self.ruling_density - np.sin(input_angle)) << u.rad

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.diffraction_order)
        out = np.broadcast(out, self.ruling_density)
        return out


@dataclasses.dataclass
class CubicPolyDensity(ConstantDensity):
    ruling_density_linear: kgpy.uncertainty.ArrayLike = 0 / (u.mm ** 2)
    ruling_density_quadratic: kgpy.uncertainty.ArrayLike = 0 / (u.mm ** 3)
    ruling_density_cubic: kgpy.uncertainty.ArrayLike = 0 / (u.mm ** 4)

    def normal(self: CubicPolyDensityT, position: kgpy.vectors.Cartesian2D) -> kgpy.vectors.Cartesian2D:
        x = position.x
        x2 = np.square(x)
        term0 = self.ruling_density
        term1 = self.ruling_density_linear * x
        term2 = self.ruling_density_quadratic * x2
        term3 = self.ruling_density_cubic * x * x2
        groove_density = term0 + term1 + term2 + term3
        return kgpy.vectors.Cartesian3D(x=groove_density, y=0 * groove_density, z=0 * groove_density)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.ruling_density_linear)
        out = np.broadcast(out, self.ruling_density_quadratic)
        out = np.broadcast(out, self.ruling_density_cubic)
        return out


@dataclasses.dataclass
class CubicPolySpacing(ConstantDensity):
    ruling_spacing_linear: kgpy.uncertainty.ArrayLike = 0 * u.dimensionless_unscaled
    ruling_spacing_quadratic: kgpy.uncertainty.ArrayLike = 0 / (u.mm ** 1)
    ruling_spacing_cubic: kgpy.uncertainty.ArrayLike = 0 / (u.mm ** 2)

    def __eq__(self: CubicPolySpacingT, other: CubicPolySpacingT) -> bool:
        if not super().__eq__(other):
            return False
        if not (other.ruling_spacing_linear == self.ruling_spacing_linear).all():
            return False
        if not (other.ruling_spacing_quadratic == self.ruling_spacing_quadratic).all():
            return False
        return True

    def normal(self: CubicPolySpacingT, position: kgpy.vectors.Cartesian2D) -> kgpy.vectors.Cartesian3D:
        x = position.x
        x2 = np.square(x)
        term0 = self.ruling_spacing
        term1 = self.ruling_spacing_linear * x
        term2 = self.ruling_spacing_quadratic * x2
        term3 = self.ruling_spacing_cubic * x * x2
        ruling_spacing = term0 + term1 + term2 + term3
        groove_density = 1 / ruling_spacing
        return kgpy.vectors.Cartesian3D(x=groove_density, y=0 * groove_density, z=0 * groove_density)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.ruling_spacing_linear)
        out = np.broadcast(out, self.ruling_spacing_quadratic)
        out = np.broadcast(out, self.ruling_spacing_cubic)
        return out

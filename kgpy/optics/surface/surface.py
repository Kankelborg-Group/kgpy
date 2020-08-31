import typing as typ
import abc
import dataclasses
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import mixin, vector, transform, optimization
from .. import Rays, Sag, Rulings, Aperture, Material

__all__ = ['Surface']

SagT = typ.TypeVar('SagT', bound=Sag)
MaterialT = typ.TypeVar('MaterialT', bound=Material)
ApertureT = typ.TypeVar('ApertureT', bound=Aperture)
ApertureMechT = typ.TypeVar('ApertureMechT', bound=Aperture)
RulingsT = typ.TypeVar('RulingsT', bound=Rulings)


@dataclasses.dataclass
class Surface(
    mixin.Broadcastable,
    transform.rigid.Transformable,
    mixin.Named,
    abc.ABC,
    typ.Generic[SagT, MaterialT, ApertureT, ApertureMechT, RulingsT],
):
    """
    Interface for representing an optical surface.
    """
    is_stop: bool = False
    is_active: bool = True
    is_visible: bool = True
    sag: typ.Optional[SagT] = None
    material: typ.Optional[MaterialT] = None
    aperture: typ.Optional[ApertureT] = None
    aperture_mechanical: typ.Optional[ApertureMechT] = None
    rulings: typ.Optional[RulingsT] = None

    def ray_intercept(
            self,
            rays: Rays,
            intercept_error: u.Quantity = 0.1 * u.nm
    ) -> u.Quantity:

        def line(t: u.Quantity) -> u.Quantity:
            return rays.position + rays.direction * t[..., None]

        def func(t: u.Quantity) -> u.Quantity:
            a = line(t)
            if self.sag is not None:
                sag = self.sag(a[vector.x], a[vector.y])
            else:
                sag = 0 * u.mm
            return a[vector.z] - sag

        bracket_max = 2 * np.nanmax(np.abs(rays.position[vector.z])) + 1 * u.mm
        # if np.isfinite(self.radius):
        #     bracket_max = np.sqrt(np.square(bracket_max) + 2 * np.square(self.radius))
        t_intercept = optimization.root_finding.false_position(
            func=func,
            bracket_min=-bracket_max,
            bracket_max=bracket_max,
            max_abs_error=intercept_error,
        )
        return line(t_intercept)

    def propagate_rays(self, rays: Rays, intercept_error: u.Quantity = 0.1 * u.nm) -> Rays:
        from_prev_to_self = rays.transform.inverse + self.transform
        from_self_to_prev = from_prev_to_self.inverse
        rays = rays.apply_transform_list(from_self_to_prev)
        rays.transform = self.transform

        rays.position = self.ray_intercept(rays, intercept_error=intercept_error)

        if self.sag is not None:
            rays.surface_normal = self.sag.normal(rays.position[vector.x], rays.position[vector.y])
        else:
            rays.surface_normal = [[0, 0, -1]] * u.dimensionless_unscaled

        if self.rulings is not None:
            a = self.rulings.effective_input_direction(rays, material=self.material)
            n1 = self.rulings.effective_input_index(rays, material=self.material)
        else:
            a = rays.direction
            n1 = rays.index_of_refraction

        if self.material is not None:
            rays.index_of_refraction = self.material.index_of_refraction(rays)
        else:
            rays.index_of_refraction = np.sign(rays.index_of_refraction) << u.dimensionless_unscaled

        r = n1 / rays.index_of_refraction

        if rays.paraxial:
            pass

        else:
            c = -vector.dot(a, rays.surface_normal)
            b = r * a + (r * c - np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * rays.surface_normal
            rays.direction = vector.normalize(b)

        if self.aperture is not None:
            if self.aperture.wire.unit.is_equivalent(u.rad):
                rays.vignetted_mask = rays.vignetted_mask & self.aperture.is_unvignetted(rays.field_angles)
            else:
                rays.vignetted_mask = rays.vignetted_mask & self.aperture.is_unvignetted(rays.position)

        return rays

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (0, 1),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        if self.is_visible:

            if self.aperture is not None:
                self.aperture.plot(ax, components, rigid_transform, self.sag, )
            if self.aperture_mechanical is not None:
                self.aperture_mechanical.plot(ax, components, rigid_transform, self.sag, )

            if self.material is not None:
                if self.aperture_mechanical is not None:
                    self.material.plot(ax, components, rigid_transform, self.sag, self.aperture_mechanical)
                elif self.aperture is not None:
                    self.material.plot(ax, components, rigid_transform, self.sag, self.aperture)

        return ax

    def copy(self) -> 'Surface[SagT, MaterialT, ApertureT, ApertureMechT, RulingsT]':
        other = super().copy()      # type: Surface[SagT, MaterialT, ApertureT, ApertureMechT, RulingsT]
        other.is_stop = self.is_stop
        other.is_active = self.is_active
        other.is_visible = self.is_visible
        other.sag = self.sag.copy()
        other.material = self.material.copy()
        other.aperture = self.aperture.copy()
        other.aperture_mechanical = self.aperture.copy()
        other.rulings = self.rulings.copy()
        return other

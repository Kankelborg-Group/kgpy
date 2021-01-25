import typing as typ
import abc
import collections
import dataclasses
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
from kgpy import mixin, vector, transform as tfrm, optimization
from ..rays import Rays, RaysList
from .sag import Sag
from .material import Material
from .aperture import Aperture
from .rulings import Rulings

__all__ = [
    'sag',
    'material',
    'aperture',
    'rulings',
    'SagT',
    'Surface',
    'SurfaceList',
]

SagT = typ.TypeVar('SagT', bound=Sag)       #: Generic :class:`kgpy.optics.surface.sag.Sag` type
MaterialT = typ.TypeVar('MaterialT', bound=Material)
ApertureT = typ.TypeVar('ApertureT', bound=Aperture)
ApertureMechT = typ.TypeVar('ApertureMechT', bound=Aperture)
RulingsT = typ.TypeVar('RulingsT', bound=Rulings)


@dataclasses.dataclass
class Surface(
    mixin.Broadcastable,
    tfrm.rigid.Transformable,
    mixin.Colorable,
    mixin.Named,
    abc.ABC,
    typ.Generic[SagT, MaterialT, ApertureT, ApertureMechT, RulingsT],
):
    """
    Interface for representing an optical surface.
    """

    is_stop: bool = False
    is_stop_test: bool = False
    is_active: bool = True  #: Flag to disable the surface
    is_visible: bool = True     #: Flag to disable plotting this surface
    sag: typ.Optional[SagT] = None      #: Sag profile of this surface
    material: typ.Optional[MaterialT] = None    #: Material type for this surface
    aperture: typ.Optional[ApertureT] = None    #: Aperture of this surface
    aperture_mechanical: typ.Optional[ApertureMechT] = None     #: Mechanical aperture of this surface
    rulings: typ.Optional[RulingsT] = None      #: Ruling profile of this surface
    # baffle_link: typ.Optional['Surface'] = None
    baffle_loft_ids: typ.List[int] = dataclasses.field(default_factory=lambda: [])

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
                surf_sag = self.sag(a[vector.x], a[vector.y])
            else:
                surf_sag = 0 * u.mm
            return a[vector.z] - surf_sag

        bracket_max = 2 * np.nanmax(np.abs(rays.position[vector.z])) + 1 * u.mm
        # if np.isfinite(self.radius):
        #     bracket_max = np.sqrt(np.square(bracket_max) + 2 * np.square(self.radius))
        t_intercept = optimization.root_finding.scalar.false_position(
            func=func,
            bracket_min=-bracket_max,
            bracket_max=bracket_max,
            max_abs_error=intercept_error,
        )
        return line(t_intercept)

    def propagate_rays(
            self,
            rays: Rays,
            intercept_error: u.Quantity = 0.1 * u.nm
    ) -> Rays:

        if not self.is_active:
            return rays

        # from_prev_to_self = rays.transform.inverse + self.transform
        # from_self_to_prev = from_prev_to_self.inverse
        # rays = rays.apply_transform_list(from_self_to_prev)
        # transform_total = rays.transform + self.transform.inverse
        transform_total = self.transform.inverse + rays.transform
        rays = rays.apply_transform_list(transform_total)
        rays.transform = self.transform

        rays.position = self.ray_intercept(rays, intercept_error=intercept_error)

        if self.rulings is not None:
            a = self.rulings.effective_input_direction(rays, material=self.material)
            n1 = self.rulings.effective_input_index(rays, material=self.material)
        else:
            a = rays.direction
            n1 = rays.index_of_refraction

        if self.material is not None:
            n2 = self.material.index_of_refraction(rays)
        else:
            n2 = np.sign(rays.index_of_refraction) << u.dimensionless_unscaled

        r = n1 / n2
        rays.index_of_refraction = n2

        if self.sag is not None:
            rays.surface_normal = self.sag.normal(rays.position[vector.x], rays.position[vector.y])
        else:
            rays.surface_normal = -vector.z_hat[None, ...] * u.dimensionless_unscaled

        c = -vector.dot(a, rays.surface_normal)
        b = r * a + (r * c - np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * rays.surface_normal
        rays.direction = vector.normalize(b)

        if self.aperture is not None:
            if self.aperture.max.unit.is_equivalent(u.rad):
                rays.vignetted_mask = rays.vignetted_mask & self.aperture.is_unvignetted(rays.field_angles)
            else:
                rays.vignetted_mask = rays.vignetted_mask & self.aperture.is_unvignetted(rays.position)

        return rays

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (0, 1),
            color: typ.Optional[str] = None,
            transform_extra: typ.Optional[tfrm.rigid.TransformList] = None,
            to_global: bool = False,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        if color is None:
            color = self.color

        if to_global:
            if transform_extra is None:
                transform_extra = tfrm.rigid.TransformList()
            transform_extra = transform_extra + self.transform

        if self.is_visible:

            if self.aperture is not None:
                self.aperture.plot(
                    ax=ax,
                    components=components,
                    transform_extra=transform_extra,
                    color=color,
                    sag=self.sag,
                )
            if self.aperture_mechanical is not None:
                self.aperture_mechanical.plot(
                    ax=ax,
                    components=components,
                    transform_extra=transform_extra,
                    color=color,
                    sag=self.sag,
                )

            if self.material is not None:
                if self.aperture_mechanical is not None:
                    self.material.plot(
                        ax=ax,
                        components=components,
                        transform_extra=transform_extra,
                        color=color,
                        sag=self.sag,
                        aperture=self.aperture_mechanical,
                    )
                elif self.aperture is not None:
                    self.material.plot(
                        ax=ax,
                        components=components,
                        transform_extra=transform_extra,
                        color=color,
                        sag=self.sag,
                        aperture=self.aperture,
                    )

        return ax

    def view(self) -> 'Surface[SagT, MaterialT, ApertureT, ApertureMechT, RulingsT]':
        other = super().view()      # type: Surface[SagT, MaterialT, ApertureT, ApertureMechT, RulingsT]
        other.is_stop = self.is_stop
        other.is_stop_test = self.is_stop_test
        other.is_active = self.is_active
        other.is_visible = self.is_visible
        other.sag = self.sag
        other.material = self.material
        other.aperture = self.aperture
        other.aperture_mechanical = self.aperture_mechanical
        other.rulings = self.rulings
        other.baffle_loft_ids = self.baffle_loft_ids
        return other

    def copy(self) -> 'Surface[SagT, MaterialT, ApertureT, ApertureMechT, RulingsT]':
        other = super().copy()      # type: Surface[SagT, MaterialT, ApertureT, ApertureMechT, RulingsT]
        other.is_stop = self.is_stop
        other.is_stop_test = self.is_stop_test
        other.is_active = self.is_active
        other.is_visible = self.is_visible

        if self.sag is None:
            other.sag = self.sag
        else:
            other.sag = self.sag.copy()

        if self.material is None:
            other.material = self.material
        else:
            other.material = self.material.copy()

        if self.aperture is None:
            other.aperture = self.aperture
        else:
            other.aperture = self.aperture.copy()

        if self.aperture_mechanical is None:
            other.aperture_mechanical = self.aperture_mechanical
        else:
            other.aperture_mechanical = self.aperture_mechanical.copy()

        if self.rulings is None:
            other.rulings = self.rulings
        else:
            other.rulings = self.rulings.copy()

        if self.baffle_loft_ids is None:
            other.baffle_loft_ids = self.baffle_loft_ids
        else:
            other.baffle_loft_ids = self.baffle_loft_ids.copy()

        return other


@dataclasses.dataclass
class SurfaceList(
    # mixin.Toleranceable,
    # collections.UserList,
    mixin.Colorable,
    tfrm.rigid.Transformable,
    mixin.DataclassList[Surface],
    # typ.List[Surface],
):

    @property
    def flat_local_iter(self) -> typ.Iterator[Surface]:
        for surf in self.data:
            if isinstance(surf, type(self)):
                for s in surf.flat_local_iter:
                    yield s
            else:
                yield surf

    @property
    def flat_global_iter(self) -> typ.Iterator[Surface]:
        for surf in self.data:
            if isinstance(surf, type(self)):
                for s in surf.flat_global_iter:
                    s = s.view()
                    s.transform = self.transform + s.transform
                    yield s
            else:
                surf = surf.view()
                surf.transform = self.transform + surf.transform
                yield surf

    @property
    def flat_global(self) -> 'SurfaceList':
        other = super().copy()  # type: SurfaceList
        other.data = list(self.flat_global_iter)
        return other

    @property
    def flat_local(self) -> 'SurfaceList':
        other = super().copy()  # type: SurfaceList
        other.data = list(self.flat_local_iter)
        return other

    def raytrace(
            self,
            rays: Rays,
            surface_last: typ.Optional[Surface] = None,
            intercept_error: u.Quantity = 0.1 * u.nm
    ) -> RaysList:

        rays_list = RaysList()
        for surf in self.flat_global_iter:
            rays = surf.propagate_rays(rays, intercept_error=intercept_error)
            rays_list.append(rays)
            if surf == surface_last:
                return rays_list
        return rays_list

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            color: typ.Optional[str] = None,
            transform_extra: typ.Optional[tfrm.rigid.TransformList] = None,
            to_global: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if color is None:
            color = self.color

        if to_global:
            if transform_extra is None:
                transform_extra = tfrm.rigid.TransformList()
            transform_extra = transform_extra + self.transform

        for surf in self:
            surf.plot(
                ax=ax,
                components=components,
                color=color,
                transform_extra=transform_extra,
                to_global=True,
            )

        return ax

    # @property
    # def tol_iter(self) -> typ.Iterator['SurfaceList']:
    #     if len(self) > 0:
    #         for s in self[0].tol_iter:
    #             result = self.view()
    #             result.data = [s]
    #             if len(self) > 1:
    #                 for slist in self[1:].tol_iter:
    #                     yield result + slist
    #             else:
    #                 yield result
    #
    #
    #     else:
    #         yield self.view()


from . import aperture
from . import sag
from . import material
from . import rulings
import typing as typ
import abc
import dataclasses
import numpy as np
import matplotlib.axes
import matplotlib.lines
import astropy.units as u
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
    mixin.Plottable,
    mixin.Named,
    abc.ABC,
    typ.Generic[SagT, MaterialT, ApertureT, ApertureMechT, RulingsT],
):
    """
    Interface for representing an optical surface.
    """
    color: str = 'black'
    is_stop: bool = False
    is_stop_test: bool = False
    is_active: bool = True  #: Flag to disable the surface
    is_visible: bool = True     #: Flag to disable plotting this surface
    sag: typ.Optional[SagT] = None      #: Sag profile of this surface
    material: typ.Optional[MaterialT] = None    #: Material type for this surface
    aperture: typ.Optional[ApertureT] = None    #: Aperture of this surface
    aperture_mechanical: typ.Optional[ApertureMechT] = None     #: Mechanical aperture of this surface
    rulings: typ.Optional[RulingsT] = None      #: Ruling profile of this surface
    baffle_loft_ids: typ.List[int] = dataclasses.field(default_factory=lambda: [])

    def ray_intercept(
            self,
            rays: Rays,
            intercept_error: u.Quantity = 0.1 * u.nm
    ) -> vector.Vector3D:

        def line(t: u.Quantity) -> vector.Vector3D:
            return rays.position + rays.direction * t

        def func(t: u.Quantity) -> u.Quantity:
            a = line(t)
            if self.sag is not None:
                surf_sag = self.sag(a.x, a.y, num_extra_dims=rays.axis.ndim)
            else:
                surf_sag = 0 * u.mm
            return a.z - surf_sag

        # bracket_max = 2 * np.nanmax(np.abs(rays.position[vector.z])) + 1 * u.mm
        # t_intercept = optimization.root_finding.scalar.false_position(
        #     func=func,
        #     bracket_min=-bracket_max,
        #     bracket_max=bracket_max,
        #     max_abs_error=intercept_error,
        # )
        t_intercept = optimization.root_finding.scalar.secant(
            func=func,
            root_guess=0 * u.mm,
            step_size=1 * u.mm,
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

        transform_total = self.transform.inverse + rays.transform
        rays = rays.apply_transform_list(transform_total)
        rays.transform = self.transform

        rays.position = self.ray_intercept(rays, intercept_error=intercept_error)

        if self.rulings is not None:
            # a = self.rulings.effective_input_direction(rays, material=self.material)
            # n1 = self.rulings.effective_input_index(rays, material=self.material)
            v = self.rulings.effective_input_vector(rays=rays, material=self.material)
            a = self.rulings.effective_input_direction(v)
            n1 = self.rulings.effective_input_index(v)
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
            rays.surface_normal = self.sag.normal(rays.position.x, rays.position.y, num_extra_dims=rays.axis.ndim)
        else:
            rays.surface_normal = -vector.z_hat.copy()

        c = -a @ rays.surface_normal
        b = r * a + (r * c - np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * rays.surface_normal
        rays.direction = b.normalize()

        if self.aperture is not None:
            if self.aperture.max.x.unit.is_equivalent(u.rad):
                new_vignetted_mask = self.aperture.is_unvignetted(rays.field_angles, num_extra_dims=rays.axis.ndim)
                rays.vignetted_mask = rays.vignetted_mask & new_vignetted_mask
            else:
                new_vignetted_mask = self.aperture.is_unvignetted(rays.position, num_extra_dims=rays.axis.ndim)
                rays.vignetted_mask = rays.vignetted_mask & new_vignetted_mask

        return rays

    def histogram(self, rays: Rays, nbins: vector.Vector2D, weights: typ.Optional[u.Quantity] = None):

        if self.aperture is not None:
            hmin = self.aperture.min
            hmax = self.aperture.max
        else:
            hmin = rays.position.min()
            hmax = rays.position.max()

        nbins = nbins.to_tuple()
        hist = np.empty(rays.shape + nbins)

        for i in range(rays.size):
            index = np.unravel_index(i, rays.shape)
            hist[index] = np.histogram2d(
                x=rays.position.x,
                y=rays.position.y,
                bins=nbins,
                range=[
                    [hmin.x, hmax.x],
                    [hmin.y, hmax.y],
                ],
                weights=weights,
            )[0]

        return hist

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            components: typ.Tuple[str, str] = ('x', 'y'),
            component_z: typ.Optional[str] = None,
            color: typ.Optional[str] = None,
            linewidth: typ.Optional[float] = None,
            linestyle: typ.Optional[str] = None,
            transform_extra: typ.Optional[tfrm.rigid.TransformList] = None,
            to_global: bool = False,
            plot_annotations: bool = True,
            annotation_text_y: float = 1.05,
    ) -> typ.List[matplotlib.lines.Line2D]:

        if color is None:
            color = self.color
        if linewidth is None:
            linewidth = self.linewidth
        if linestyle is None:
            linestyle = self.linestyle

        if transform_extra is None:
            transform_extra = tfrm.rigid.TransformList()

        if to_global:
            transform_extra = transform_extra + self.transform

        lines = []

        if self.is_visible:

            kwargs = dict(
                ax=ax,
                components=components,
                component_z=component_z,
                transform_extra=transform_extra,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                sag=self.sag,
            )

            if self.aperture is not None:
                lines += self.aperture.plot(**kwargs)
            if self.aperture_mechanical is not None:
                lines += self.aperture_mechanical.plot(**kwargs)

            if self.material is not None:
                if self.aperture_mechanical is not None:
                    lines += self.material.plot(**kwargs, aperture=self.aperture_mechanical, )
                elif self.aperture is not None:
                    lines += self.material.plot(**kwargs, aperture=self.aperture, )

            if plot_annotations:
                c_x, c_y = components
                text_position_local = vector.Vector3D.spatial().reshape(-1)
                if self.aperture_mechanical is not None:
                    if self.aperture_mechanical.max.x.unit.is_equivalent(u.mm):
                        text_position_local = self.aperture_mechanical.wire
                elif self.aperture is not None:
                    if self.aperture.max.x.unit.is_equivalent(u.mm):
                        text_position_local = self.aperture.wire

                text_position = transform_extra(text_position_local, num_extra_dims=1)
                text_position = text_position.reshape(-1, text_position.shape[~0])

                for i in range(1):

                    wire_index = np.argmax(text_position[i].get_component(c_y))

                    text_x = text_position[i][wire_index].get_component(c_x)
                    text_y = text_position[i][wire_index].get_component(c_y)

                    ax.annotate(
                        text=self.name,
                        xy=(text_x, text_y),
                        xytext=(text_x, annotation_text_y),
                        # textcoords='offset points',
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        rotation=60,
                        textcoords=ax.get_xaxis_transform(),
                        arrowprops=dict(
                            color='black',
                            width=0.5,
                            headwidth=4,
                            alpha=0.5,
                        ),
                    )

        return lines

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
    mixin.Colorable,
    tfrm.rigid.Transformable,
    mixin.DataclassList[Surface],
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
            ax: matplotlib.axes.Axes,
            components: typ.Tuple[str, str] = ('x', 'y'),
            component_z: typ.Optional[str] = None,
            color: typ.Optional[str] = None,
            linewidth: typ.Optional[float] = None,
            linestyle: typ.Optional[str] = None,
            transform_extra: typ.Optional[tfrm.rigid.TransformList] = None,
            to_global: bool = False,
            plot_annotations: bool = True,
            annotation_text_y: float = 1.05,
    ) -> typ.List[matplotlib.lines.Line2D]:

        if color is None:
            color = self.color

        if transform_extra is None:
            transform_extra = tfrm.rigid.TransformList()

        # if to_global:
        #     transform_extra = transform_extra + self.transform

        lines = []
        for surf in self:
            lines += surf.plot(
                ax=ax,
                components=components,
                component_z=component_z,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                transform_extra=transform_extra,
                to_global=True,
                plot_annotations=plot_annotations,
                annotation_text_y=annotation_text_y,
            )

        return lines


from . import aperture
from . import sag
from . import material
from . import rulings
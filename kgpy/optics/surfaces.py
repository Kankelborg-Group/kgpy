import typing as typ
import abc
import dataclasses
import numpy as np
import matplotlib.axes
import matplotlib.lines
import astropy.units as u
import astropy.visualization
from ezdxf.addons.r12writer import R12FastStreamWriter
import kgpy.mixin
import kgpy.uncertainty
import kgpy.vectors
import kgpy.transforms
import kgpy.optimization
import kgpy.io.dxf
from . import sags
from . import apertures
from . import rays
from . import materials
from . import rulings

__all__ = [
    'sags',
    'materials',
    'apertures',
    'rulings',
    'SagT',
    'Surface',
    'SurfaceList',
]

SurfaceT = typ.TypeVar('SurfaceT', bound='Surface')
SurfaceListT = typ.TypeVar('SurfaceListT', bound='SurfaceList')
SagT = typ.TypeVar('SagT', bound='sags.Sag')       #: Generic :class:`kgpy.optics.surface.sag.Sag` type
MaterialT = typ.TypeVar('MaterialT', bound='materials.Material')
ApertureT = typ.TypeVar('ApertureT', bound='apertures.Aperture')
ApertureMechT = typ.TypeVar('ApertureMechT', bound='apertures.Aperture')
RulingT = typ.TypeVar('RulingT', bound='rulings.Ruling')


@dataclasses.dataclass(eq=False)
class Surface(
    kgpy.io.dxf.WritableMixin,
    kgpy.mixin.Broadcastable,
    kgpy.transforms.Transformable,
    kgpy.mixin.Plottable,
    kgpy.mixin.Named,
    abc.ABC,
    typ.Generic[SagT, MaterialT, ApertureT, ApertureMechT, RulingT],
):
    """
    Interface for representing an optical surface.
    """
    plot_kwargs: typ.Optional[typ.Dict[str, typ.Any]] = dataclasses.field(default_factory=lambda: dict(color='black'))
    is_field_stop: bool = False
    is_pupil_stop: bool = False
    is_pupil_stop_test: bool = False
    is_active: bool = True  #: Flag to disable the surface
    is_visible: bool = True     #: Flag to disable plotting this surface
    sag: typ.Optional[SagT] = None      #: Sag profile of this surface
    material: typ.Optional[MaterialT] = None    #: Material type for this surface
    aperture: typ.Optional[ApertureT] = None    #: Aperture of this surface
    aperture_mechanical: typ.Optional[ApertureMechT] = None     #: Mechanical aperture of this surface
    ruling: typ.Optional[RulingT] = None      #: Ruling profile of this surface
    baffle_loft_ids: typ.List[int] = dataclasses.field(default_factory=lambda: [])

    def __eq__(self: SurfaceT, other: SurfaceT):
        if not isinstance(other, type(self)):
            return False
        if not super().__eq__(other):
            return False
        if not self.is_field_stop == other.is_field_stop:
            return False
        if not self.is_pupil_stop == other.is_pupil_stop:
            return False
        if not self.is_pupil_stop_test == other.is_pupil_stop_test:
            return False
        if not self.is_active == other.is_active:
            return False
        if not self.is_visible == other.is_visible:
            return False
        if not self.sag == other.sag:
            return False
        if not self.material == other.material:
            return False
        if not self.aperture == other.aperture:
            return False
        if not self.aperture_mechanical == other.aperture_mechanical:
            return False
        if not self.ruling == other.ruling:
            return False
        if not self.baffle_loft_ids == other.baffle_loft_ids:
            return False
        return True

    def ray_intercept(
            self,
            ray: rays.RayVector,
            intercept_error: u.Quantity = 0.1 * u.nm,
    ) -> kgpy.vectors.Cartesian3D:

        def line(t: kgpy.uncertainty.ArrayLike) -> kgpy.vectors.Cartesian3D:
            return ray.position + ray.direction * t

        def func(t: kgpy.uncertainty.ArrayLike) -> kgpy.uncertainty.ArrayLike:
            a = line(t)
            if self.sag is not None:
                surf_sag = self.sag(a)
            else:
                surf_sag = 0 * u.m
            return a.z - surf_sag

        t_intercept = kgpy.optimization.root_finding.secant(
            func=func,
            root_guess=0 * u.mm,
            step_size=1 * u.mm,
            max_abs_error=intercept_error,
        )
        return line(t_intercept)

    def propagate_rays(
            self,
            ray: rays.RayVector,
            intercept_error: u.Quantity = 0.1 * u.nm
    ) -> rays.RayVector:

        ray = ray.copy_shallow()

        if not self.is_active:
            return ray

        transform_total = self.transform.inverse + ray.transform
        ray = ray.apply_transform(transform_total)
        ray.transform = self.transform

        ray.position = self.ray_intercept(ray, intercept_error=intercept_error)

        if self.ruling is not None:
            v = self.ruling.effective_input_vector(ray=ray, material=self.material)
            a = self.ruling.effective_input_direction(v)
            n1 = self.ruling.effective_input_index(v)
        else:
            a = ray.direction
            n1 = ray.index_refraction

        if self.material is not None:
            n2 = self.material.index_refraction(ray)
        else:
            n2 = np.sign(ray.index_refraction) << u.dimensionless_unscaled

        r = n1 / n2
        ray.index_refraction = n2

        if self.sag is not None:
            ray.surface_normal = self.sag.normal(ray.position)
        else:
            ray.surface_normal = -kgpy.vectors.Cartesian3D.z_hat()

        c = -a @ ray.surface_normal
        b = r * a + (r * c - np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * ray.surface_normal
        ray.direction = b.normalized

        if self.material is not None:
            ray.intensity = self.material.transmissivity(ray) * ray.intensity

        if self.aperture is not None:
            if self.aperture.max.x.unit.is_equivalent(u.rad):
                ray.mask = ray.mask & self.aperture.is_unvignetted(ray.field_angles)
            else:
                ray.mask = ray.mask & self.aperture.is_unvignetted(ray.position)

        return ray

    # def histogram(self, rays: Rays, nbins: vector.Vector2D, weights: typ.Optional[u.Quantity] = None):
    #
    #     if self.aperture is not None:
    #         hmin = self.aperture.min
    #         hmax = self.aperture.max
    #     else:
    #         hmin = rays.position.min()
    #         hmax = rays.position.max()
    #
    #     nbins = nbins.to_tuple()
    #     hist = np.empty(rays.shape + nbins)
    #
    #     for i in range(rays.size):
    #         index = np.unravel_index(i, rays.shape)
    #         hist[index] = np.histogram2d(
    #             x=rays.position.x,
    #             y=rays.position.y,
    #             bins=nbins,
    #             range=[
    #                 [hmin.x, hmax.x],
    #                 [hmin.y, hmax.y],
    #             ],
    #             weights=weights,
    #         )[0]
    #
    #     return hist

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            component_x: str = 'x',
            component_y: str = 'y',
            component_z: str = 'z',
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            to_global: bool = False,
            plot_annotations: bool = True,
            annotation_text_y: float = 1.05,
            **kwargs,
    ) -> typ.List[matplotlib.lines.Line2D]:

        kwargs = {**self.plot_kwargs, **kwargs}

        if transform_extra is None:
            transform_extra = kgpy.transforms.TransformList()

        if to_global:
            transform_extra = transform_extra + self.transform

        lines = []

        if self.is_visible:

            kwargs = dict(
                ax=ax,
                component_x=component_x,
                component_y=component_y,
                component_z=component_z,
                transform_extra=transform_extra,
                sag=self.sag,
                **kwargs
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
                text_position_local = kgpy.vectors.Cartesian3D() * u.mm
                if self.aperture_mechanical is not None:
                    if self.aperture_mechanical.max.x.unit.is_equivalent(u.mm):
                        text_position_local = self.aperture_mechanical.wire
                elif self.aperture is not None:
                    if self.aperture.max.x.unit.is_equivalent(u.mm):
                        text_position_local = self.aperture.wire

                text_position = transform_extra(text_position_local)
                # text_position = text_position.reshape(-1, text_position.shape[~0])

                for index in text_position.ndindex(axis_ignored='wire'):
                    text_pos = text_position[index]

                    wire_index = np.argmax(text_pos.coordinates[component_y], axis='wire')

                    text_pos_max = text_pos[dict(wire=wire_index)]

                    text_x = text_pos_max.coordinates[component_x]
                    text_y = text_pos_max.coordinates[component_y]

                    with astropy.visualization.quantity_support():
                        ax.annotate(
                            text=self.name,
                            xy=(text_x, text_y),
                            xytext=(text_x, annotation_text_y),
                            # textcoords='offset points',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            rotation='vertical',
                            textcoords=ax.get_xaxis_transform(),
                            arrowprops=dict(
                                arrowstyle='-|>',
                                linewidth=0.5,
                                color='red',
                                relpos=(0, 0)
                                # width=0.5,
                                # headwidth=4,
                                # alpha=0.5,
                            ),
                        )

        return lines

    def write_to_dxf(
            self: SurfaceT,
            file_writer: R12FastStreamWriter,
            unit: u.Unit,
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
    ) -> None:
        if transform_extra is None:
            transform_extra = kgpy.transforms.TransformList()

        transform_extra = transform_extra + self.transform

        super().write_to_dxf(
            file_writer=file_writer,
            unit=unit,
            transform_extra=transform_extra,
        )
        if self.aperture is not None:
            self.aperture.write_to_dxf(
                file_writer=file_writer,
                unit=unit,
                transform_extra=transform_extra,
                sag=self.sag,
            )
        if self.aperture_mechanical is not None:
            self.aperture_mechanical.write_to_dxf(
                file_writer=file_writer,
                unit=unit,
                transform_extra=transform_extra,
                sag=self.sag,
            )


@dataclasses.dataclass
class SurfaceList(
    kgpy.io.dxf.WritableMixin,
    # mixin.Colorable,
    kgpy.mixin.Plottable,
    kgpy.transforms.Transformable,
    kgpy.mixin.DataclassList[Surface],
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
                    s = s.copy_shallow()
                    s.transform = self.transform + s.transform
                    yield s
            else:
                surf = surf.copy_shallow()
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
            ray_function: rays.RayFunction,
            surface_last: typ.Optional[Surface] = None,
            intercept_error: u.Quantity = 0.1 * u.nm
    ) -> rays.RayFunctionList:

        ray_function_list = rays.RayFunctionList()
        for surf in self.flat_global_iter:
            ray_function = ray_function.copy_shallow()
            ray_function.output = surf.propagate_rays(ray_function.output, intercept_error=intercept_error)
            ray_function_list.append(ray_function)
            if surf == surface_last:
                return ray_function_list
        return ray_function_list

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            component_x: str = 'x',
            component_y: str = 'y',
            component_z: str = 'z',
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            to_global: bool = False,
            plot_annotations: bool = True,
            annotation_text_y: float = 1.05,
            **kwargs,
    ) -> typ.List[matplotlib.lines.Line2D]:

        kwargs = {**self.plot_kwargs, **kwargs}

        if transform_extra is None:
            transform_extra = kgpy.transforms.TransformList()

        # if to_global:
        #     transform_extra = transform_extra + self.transform

        lines = []
        for surf in self:
            lines += surf.plot(
                ax=ax,
                component_x=component_x,
                component_y=component_y,
                component_z=component_z,
                transform_extra=transform_extra,
                to_global=True,
                plot_annotations=plot_annotations,
                annotation_text_y=annotation_text_y,
                **kwargs
            )

        return lines

    def write_to_dxf(
            self: SurfaceListT,
            file_writer: R12FastStreamWriter,
            unit: u.Unit,
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
    ) -> None:

        for surf in self:
            surf.write_to_dxf(
                file_writer=file_writer,
                unit=unit,
                transform_extra=transform_extra,
            )

import typing as typ
import abc
import dataclasses
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.axes
import astropy.units as u
import astropy.constants
import astropy.visualization
import astropy.modeling
from ezdxf.addons.r12writer import R12FastStreamWriter
import kgpy.plot
import kgpy.mixin
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.function
import kgpy.transforms
import kgpy.format
import kgpy.io.dxf
from . import vectors
from . import aberrations
# from .aberration import Distortion, Vignetting, Aberration

__all__ = [
    'Axis',
    # 'RayGrid',
    'RayVector',
    'RayFunction',
    'RayFunctionList',
]


RayVectorT = typ.TypeVar('RayVectorT', bound='RayVector')
RaysT = typ.TypeVar('RaysT', bound='Rays')
RaysListT = typ.TypeVar('RaysListT', bound='RaysList')


class Axis(kgpy.mixin.AutoAxis):
    ndim_pupil: typ.ClassVar[int] = 2
    ndim_field: typ.ClassVar[int] = 2

    def __init__(self):
        super().__init__()
        self.velocity_los = self.auto_axis_index()
        self.wavelength = self.auto_axis_index()
        self.pupil_y = self.auto_axis_index()
        self.pupil_x = self.auto_axis_index()
        self.field_y = self.auto_axis_index()
        self.field_x = self.auto_axis_index()
        # self.wavelength = self.auto_axis_index()

    @property
    def pupil_xy(self) -> typ.Tuple[int, int]:
        return self.pupil_x, self.pupil_y

    @property
    def field_xy(self) -> typ.Tuple[int, int]:
        return self.field_x, self.field_y

    @property
    def latex_names(self) -> typ.List[str]:
        names = [None] * self.ndim
        names[self.field_x] = 'field $x$'
        names[self.field_y] = 'field $y$'
        names[self.pupil_x] = 'pupil $x$'
        names[self.pupil_y] = 'pupil $y$'
        names[self.wavelength] = 'wavelength'
        names[self.velocity_los] = 'LOS velocity'
        return names


# @dataclasses.dataclass
# class RayGrid(
#     kgpy.mixin.Copyable,
#     abc.ABC,
# ):
#     axis: typ.ClassVar[Axis] = Axis()
#     field: kgpy.grid.RegularGrid2D = dataclasses.field(default_factory=kgpy.grid.RegularGrid2D)
#     pupil: kgpy.grid.RegularGrid2D = dataclasses.field(default_factory=kgpy.grid.RegularGrid2D)
#     wavelength: kgpy.grid.Grid1D = dataclasses.field(default_factory=lambda: kgpy.grid.RegularGrid1D(min=0 * u.nm, max=0 * u.nm))
#     velocity_los: kgpy.grid.Grid1D = dataclasses.field(
#         default_factory=lambda: kgpy.grid.RegularGrid1D(min=0 * u.km / u.s, max=0 * u.km / u.s)
#     )
#
#     @property
#     def shape(self) -> typ.Tuple[int, ...]:
#         return np.broadcast(
#             np.expand_dims(self.field.points.x, self.axis.perp_axes(self.axis.field_x)),
#             np.expand_dims(self.field.points.y, self.axis.perp_axes(self.axis.field_y)),
#             np.expand_dims(self.pupil.points.x, self.axis.perp_axes(self.axis.pupil_x)),
#             np.expand_dims(self.pupil.points.y, self.axis.perp_axes(self.axis.pupil_y)),
#             np.expand_dims(self.wavelength.points, self.axis.perp_axes(self.axis.wavelength)),
#             np.expand_dims(self.velocity_los.points, self.axis.perp_axes(self.axis.velocity_los)),
#         ).shape
#
#     @property
#     def points_field(self) -> kgpy.vector.Vector2D:
#         return self.field.mesh(shape=self.shape, new_axes=self.axis.perp_axes([self.axis.field_x, self.axis.field_y]))
#
#     @property
#     def points_pupil(self) -> kgpy.vector.Vector2D:
#         return self.pupil.mesh(shape=self.shape, new_axes=self.axis.perp_axes([self.axis.pupil_x, self.axis.pupil_y]))
#
#     @property
#     def points_wavelength(self) -> u.Quantity:
#         return self.wavelength.mesh(shape=self.shape, new_axes=self.axis.perp_axes(self.axis.wavelength))
#
#     @property
#     def points_velocity_los(self) -> u.Quantity:
#         return self.velocity_los.mesh(shape=self.shape, new_axes=self.axis.perp_axes(self.axis.velocity_los))
#
#     def points(self, component_axis: int = ~0) -> u.Quantity:
#
#         points_field = self.points_field
#         points_pupil = self.points_pupil
#
#         p = [None] * self.axis.ndim
#         p[self.axis.field_x] = points_field.x
#         p[self.axis.field_y] = points_field.y
#         p[self.axis.pupil_x] = points_pupil.x
#         p[self.axis.pupil_y] = points_pupil.y
#         p[self.axis.wavelength] = self.points_wavelength
#         p[self.axis.velocity_los] = self.points_velocity_los
#         return np.stack(arrays=p, axis=component_axis)
#
#     # @property
#     # def grids(self) -> typ.List[u.Quantity]:
#     #     return [
#     #         self.field.points.x,
#     #         self.field.points.y,
#     #         self.pupil.points.x,
#     #         self.pupil.points.y,
#     #         self.wavelength.points,
#     #         self.velocity_los.points,
#     #     ]
#
#     def points_from_axis(self, axis: int):
#         if axis == self.axis.field_x:
#             return self.points_field.x
#         elif axis == self.axis.field_y:
#             return self.points_field.y
#         elif axis == self.axis.pupil_x:
#             return self.points_pupil.x
#         elif axis == self.axis.pupil_y:
#             return self.points_pupil.y
#         elif axis == self.axis.wavelength:
#             return self.points_wavelength
#         elif axis == self.axis.velocity_los:
#             return self.points_velocity_los
#         else:
#             raise ValueError('Unsupported axis')




@dataclasses.dataclass(eq=False)
class RayVector(
    kgpy.transforms.Transformable,
    kgpy.vectors.AbstractVector,
):
    intensity: kgpy.uncertainty.ArrayLike = 1 * u.dimensionless_unscaled
    position: kgpy.vectors.Cartesian3D = dataclasses.field(default_factory=lambda: kgpy.vectors.Cartesian3D() * u.mm)
    direction: kgpy.vectors.Cartesian3D = dataclasses.field(default_factory=lambda: kgpy.vectors.Cartesian3D.z_hat() * u.dimensionless_unscaled)
    polarization: vectors.StokesVector = dataclasses.field(default_factory=vectors.StokesVector)
    wavelength: kgpy.uncertainty.ArrayLike = 0 * u.nm
    index_refraction: kgpy.uncertainty.ArrayLike = 1 * u.dimensionless_unscaled
    surface_normal: kgpy.vectors.Cartesian3D = dataclasses.field(default_factory=lambda: -kgpy.vectors.Cartesian3D.z_hat() * u.dimensionless_unscaled)
    mask: kgpy.uncertainty.ArrayLike = dataclasses.field(default_factory=lambda: kgpy.labeled.Array(True))

    @classmethod
    def from_field_angles(
            cls: typ.Type[RayVectorT],
            scene_vector: vectors.ObjectVector,
            position: kgpy.vectors.Cartesian3D,
    ):
        angle_x = -scene_vector.field.x
        angle_y = scene_vector.field.y

        transform = kgpy.transforms.TransformList([
            kgpy.transforms.RotationY(angle_x),
            kgpy.transforms.RotationX(angle_y),
        ])

        return cls(
            position=position,
            direction=transform(kgpy.vectors.Cartesian3D.z_hat()),
            wavelength=scene_vector.wavelength_doppler,
        )

    @property
    def angles(self: RayVectorT) -> kgpy.vectors.Cartesian2D:
        direction = self.direction
        return kgpy.vectors.Cartesian2D(
            x=-np.arctan2(direction.x, direction.z),
            y=np.arcsin(direction.y / direction.length),
        )

    @angles.setter
    def angles(self: RayVectorT, value: kgpy.vectors.Cartesian2D):
        transform = kgpy.transforms.TransformList([
            kgpy.transforms.RotationY(-value.x),
            kgpy.transforms.RotationX(value.y),
        ])
        self.direction = transform(kgpy.vectors.Cartesian3D.z_hat(), translate=False)


    # @property
    # def direction_cos(self: RayVectorT) -> kgpy.vector.Cartesian3D:
    #     direction = self.direction
    #     a = -direction.x
    #     b = direction.y
    #     return kgpy.vector.Cartesian3D(
    #         x=np.sin(a) * np.cos(b),
    #         y=-np.sin(b),
    #         z=np.cos(a) * np.cos(b),
    #     )
    #
    # @direction_cos.setter
    # def direction_cos(self: RayVectorT, value: kgpy.vector.Cartesian3D) -> None:
    #     self.direction.x = -np.arctan2(value.x, value.z)
    #     self.direction.y = np.arccos(value.y / value.length)

    def apply_transform(self, transform: kgpy.transforms.TransformList) -> RayVectorT:
        # other = self.copy()
        other = self.copy_shallow()
        other.position = transform(other.position)
        other.direction = transform(other.direction, translate=False)
        other.surface_normal = transform(other.surface_normal, translate=False)
        other.transform = other.transform + transform.inverse
        return other

    @property
    def transformed(self) -> RayVectorT:
        other = self.apply_transform(self.transform)
        other.transform = kgpy.transforms.TransformList()
        return other

    @property
    def energy(self) -> kgpy.uncertainty.ArrayLike:
        return (astropy.constants.h * astropy.constants.c / self.wavelength).to(u.eV)

    def _calc_average_pupil(self, a: kgpy.vectors.Cartesian3D) -> kgpy.vectors.Cartesian3D:
        return np.mean(a=a, axis=('pupil.x', 'pupil.y'), where=self.mask)

    def _calc_relative_pupil(self, a: kgpy.vectors.Cartesian3D) -> kgpy.vectors.Cartesian3D:
        return a - self._calc_average_pupil(a)

    @property
    def position_average_pupil(self) -> kgpy.vectors.Cartesian3D:
        return self._calc_average_pupil(self.position)

    @property
    def position_relative_pupil(self) -> kgpy.vectors.Cartesian3D:
        return self._calc_relative_pupil(self.position)

    @property
    def spot_size_rms(self):
        position = self.position_relative_pupil
        r = position.xy.length
        result = np.sqrt(np.mean(np.square(r), axis=('pupil_x', 'pupil_y'), where=self.mask))
        result[~self.mask.any(axis=('pupil_x', 'pupil_y'))] = np.nan
        return result


@dataclasses.dataclass
class RayFunction(
    kgpy.function.Array[vectors.ObjectVector, RayVector],
):
    input: vectors.ObjectVector = dataclasses.field(default_factory=vectors.ObjectVector)
    output: RayVector = dataclasses.field(default_factory=RayVector)

    # axis = Axis()
    #
    # intensity: u.Quantity = 1 * u.dimensionless_unscaled
    # wavelength: u.Quantity = 0 * u.nm
    # position: kgpy.vector.Cartesian3D = dataclasses.field(default_factory=lambda: kgpy.vector.Cartesian3D() * u.mm)
    # direction: kgpy.vector.Vector3D = dataclasses.field(default_factory=kgpy.vector.zhat_factory)
    # velocity_los: u.Quantity = 0 * u.km / u.s
    # surface_normal: kgpy.vector.Vector3D = dataclasses.field(default_factory=lambda: -kgpy.vector.zhat_factory())
    # index_of_refraction: u.Quantity = 1 * u.dimensionless_unscaled
    # vignetted_mask: np.ndarray = np.array([True])
    # error_mask: np.ndarray = np.array([True])
    # input_grid: typ.Optional[RayGrid] = None
    distortion_polynomial_degree: int = 2
    vignetting_polynomial_degree: int = 1
    #
    # @property
    # def field_angles(self) -> kgpy.vector.Vector2D:
    #     angle = np.arcsin(self.direction.xy).to(u.deg)
    #     angle.y = -angle.y
    #     return angle
    #
    # @classmethod
    # def from_field_angles(
    #         cls,
    #         # wavelength_grid: u.Quantity,
    #         input_grid: RayGrid,
    #         position: kgpy.vector.Vector3D,
    #         # field_grid: vector.Vector2D,
    #         # pupil_grid: vector.Vector2D,
    #         # velocity_z_grid: u.Quantity
    # ) -> 'Rays':
    #
    #     # field_x = np.expand_dims(input_grid.field.points.x, cls.axis.perp_axes(cls.axis.field_x))
    #     # field_y = np.expand_dims(input_grid.field.points.y, cls.axis.perp_axes(cls.axis.field_y))
    #
    #     direction = kgpy.transform.rigid.TiltX(input_grid.points_field.y)(kgpy.vector.z_hat)
    #     direction = kgpy.transform.rigid.TiltY(input_grid.points_field.x)(direction)
    #
    #     return cls(
    #         wavelength=input_grid.points_wavelength,
    #         position=position,
    #         direction=direction,
    #         velocity_los=input_grid.points_velocity_los,
    #         input_grid=input_grid,
    #         # input_wavelength=wavelength_grid,
    #         # input_field=field_grid,
    #         # input_pupil=pupil_grid,
    #         # input_velocity_z=velocity_z_grid,
    #     )
    #
    # @classmethod
    # def from_field_positions(
    #         cls,
    #         # intensity: u.Quantity,
    #         # wavelength_grid: u.Quantity,
    #         input_grid: RayGrid,
    #         direction: kgpy.vector.Vector3D,
    #         # field_grid: vector.Vector2D,
    #         # pupil_grid: vector.Vector2D,
    #         # velocity_z_grid: u.Quantity,
    # ) -> 'Rays':
    #
    #     return cls(
    #         wavelength=input_grid.points_wavelength,
    #         position=input_grid.points_field.to_3d(z=0 * u.mm),
    #         direction=direction,
    #         velocity_los=input_grid.points_velocity_los,
    #         input_grid=input_grid,
    #         # input_wavelength=wavelength_grid,
    #         # input_field=field_grid,
    #         # input_pupil=pupil_grid,
    #         # input_velocity_z=velocity_z_grid,
    #     )
    #
    # def apply_transform_list(self, transform_list: kgpy.transform.rigid.TransformList) -> 'Rays':
    #     # other = self.copy()
    #     other = self.copy_shallow()
    #     transform_list = transform_list.simplified
    #     other.position = transform_list(other.position, num_extra_dims=self.axis.ndim)
    #     other.direction = transform_list(other.direction, translate=False, num_extra_dims=self.axis.ndim)
    #     other.surface_normal = transform_list(other.surface_normal, translate=False, num_extra_dims=self.axis.ndim)
    #     other.transform = self.transform + transform_list.inverse
    #     return other
    #
    # @property
    # def transformed(self) -> 'Rays':
    #     other = self.apply_transform_list(self.transform)
    #     other.transform = kgpy.transform.rigid.TransformList()
    #     return other
    #
    # @property
    # def grid_shape(self) -> typ.Tuple[int, ...]:
    #     return np.broadcast(
    #         self.wavelength,
    #         self.position,
    #         self.direction,
    #         self.mask,
    #     ).shape
    #
    # @property
    # def shape(self) -> typ.Tuple[int, ...]:
    #     return self.grid_shape[:~(self.axis.ndim - 1)]
    #
    # @property
    # def base_shape(self):
    #     return self.grid_shape[~(self.axis.ndim - 1)]
    #
    # @property
    # def ndim(self):
    #     return len(self.shape)
    #
    # @property
    # def size(self) -> int:
    #     return int(np.prod(np.array(self.shape)))
    #
    # @property
    # def num_wavlength(self):
    #     return self.grid_shape[self.axis.wavelength]
    #
    # @property
    # def mask(self) -> np.ndarray:
    #     return self.vignetted_mask & self.error_mask

    @property
    def output_position_apparent(self):
        position = self.position.copy()
        position.z = np.broadcast_to(self.wavelength, position.shape, subok=True).copy()
        position = self.distortion.model(inverse=True)(position)
        return position

    # @property
    # def distortion(self) -> Distortion:
    #     return Distortion(
    #
    #     )
    #     return Distortion(
    #         wavelength=self.input_grid.wavelength.points[..., np.newaxis, np.newaxis, :],
    #         spatial_mesh_input=kgpy.vector.Vector2D(
    #             x=self.input_grid.field.points.x[..., :, np.newaxis, np.newaxis],
    #             y=self.input_grid.field.points.y[..., np.newaxis, :, np.newaxis],
    #         ),
    #         spatial_mesh_output=self.position_avg_pupil[..., 0, 0, :, 0].xy,
    #         mask=self.mask.any((self.axis.pupil_x, self.axis.pupil_y, self.axis.velocity_los)),
    #         polynomial_degree=self.distortion_polynomial_degree,
    #     )
    #
    # @property
    # def vignetting(self) -> Vignetting:
    #     intensity = self.intensity.copy()
    #     intensity, mask = np.broadcast_arrays(intensity, self.mask, subok=True)
    #     intensity[~mask] = 0
    #     counts = intensity.sum((self.axis.pupil_x, self.axis.pupil_y, self.axis.velocity_los))
    #     return Vignetting(
    #         wavelength=self.input_grid.wavelength.points[..., np.newaxis, np.newaxis, :],
    #         spatial_mesh=kgpy.vector.Vector2D(
    #             x=self.input_grid.field.points.x[..., :, np.newaxis, np.newaxis],
    #             y=self.input_grid.field.points.y[..., np.newaxis, :, np.newaxis],
    #         ),
    #         unvignetted_percent=counts,
    #         mask=mask.any((self.axis.pupil_x, self.axis.pupil_y, self.axis.velocity_los)),
    #         polynomial_degree=self.vignetting_polynomial_degree,
    #     )
    #
    # @property
    # def aberration(self) -> Aberration:
    #     return Aberration(
    #         distortion=self.distortion,
    #         vignetting=self.vignetting,
    #     )

    # @property
    # def spot_size_rms(self):
    #     position = self.position_relative_pupil
    #     r = position.xy.length
    #     r2 = np.square(r)
    #     pupil_axes = self.axis.pupil_x, self.axis.pupil_y
    #     sz = np.sqrt(np.ma.average(r2.value, axis=pupil_axes, weights=self.mask) << r2.unit)
    #     mask = self.mask.any(pupil_axes)
    #     sz[~mask] = np.nan
    #     return sz

    def plot_spot_size_vs_field(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
            velocity_los_index: int = 0,
            kwargs_colorbar: typ.Optional[typ.Dict[str, typ.Any]] = None,
            digits_after_decimal: int = 3,
    ) -> typ.MutableSequence[plt.Axes]:
        if axs is None:
            fig, axs = plt.subplots(ncols=self.num_wavlength)
        else:
            fig = axs[0].figure

        if kwargs_colorbar is None:
            kwargs_colorbar = {}

        wavelength = self.input_grid.wavelength.points
        wavelength_name = self.input_grid.wavelength.name
        field_x, field_y = self.input_grid.field.points.to_tuple()
        sizes = self.spot_size_rms

        sl = [slice(None)] * sizes.ndim
        wsl = slice(None, len(axs))
        sl[self.axis.wavelength] = wsl
        wavelength = wavelength[..., wsl]
        wavelength_name = wavelength_name[..., wsl]
        sizes = sizes[sl]

        if config_index is not None:
            field_x, field_y = field_x[config_index], field_y[config_index]
            wavelength = wavelength[config_index]
            sizes = sizes[config_index]

        sorted_indices = np.argsort(wavelength)
        sorted_slice = [slice(None)] * sizes.ndim
        sorted_slice[self.axis.wavelength] = sorted_indices
        wavelength = wavelength[sorted_indices]
        wavelength_name = wavelength_name[sorted_indices]
        sizes = sizes[sorted_slice]

        vmin, vmax = np.nanmin(sizes), np.nanmax(sizes)

        # for ax, wavl, sz in zip(axs, wavelength, sizes):
        for i in range(len(axs)):
            wavelength_formatted = kgpy.format.quantity(wavelength[i], digits_after_decimal=digits_after_decimal)
            axs[i].set_title(f'{wavelength_name[i]} {wavelength_formatted}')
            sl = [slice(None)] * sizes.ndim
            sl[self.axis.wavelength] = i
            sl[self.axis.velocity_los] = velocity_los_index

            extent = kgpy.plot.calc_extent(
                data_min=kgpy.vectors.Vector2D(field_x.min(), field_y.min()),
                data_max=kgpy.vectors.Vector2D(field_x.max(), field_y.max()),
                num_steps=kgpy.vectors.Vector2D.from_quantity(sizes[sl].shape * u.dimensionless_unscaled),
            )

            img = axs[i].imshow(
                X=sizes[sl].T.value,
                vmin=vmin.value,
                vmax=vmax.value,
                origin='lower',
                extent=[e.value for e in extent],
            )
            axs[i].set_xlabel('input $x$ ' + '(' + "{0:latex}".format(field_x.unit) + ')')

        axs[0].set_ylabel('input $y$ ' + '(' + "{0:latex}".format(field_y.unit) + ')')

        fig.colorbar(
            img,
            ax=axs,
            label='RMS spot radius (' + '{0:latex}'.format(sizes.unit) + ')',
            **kwargs_colorbar,
        )

        return axs

    def psf(
            self,
            bins: typ.Optional[kgpy.vectors.Cartesian2D] = None,
            use_vignetted: bool = False,
            use_position_relative: bool = True,
            use_position_apparent: bool = False,
    ):
        if use_position_apparent:
            position = self.output_position_apparent
        else:
            position = self.output.position
        position = position.xy

        if not use_vignetted:
            mask = self.output.mask
        else:
            mask = kgpy.labeled.Array(True)

        if use_position_relative:
            position = position - position.mean(axis=('pupil.x', 'pupil.y'), where=mask)

        if bins is None:
            bins = kgpy.vectors.Cartesian2D(
                x=position.shape['pupil.x'],
                y=position.shape['pupil.y']
            )

        bins_dict = dict()
        bins_dict['pupil.x'] = bins.x
        bins_dict['pupil.y'] = bins.y

        axis = ('field.x', 'field.y', 'pupil.x', 'pupil.y', 'wavelength', 'velocity_los', '_distribution')

        limit_min = position.min(axis=axis, where=mask, initial=0)
        limit_max = position.max(axis=axis, where=mask, initial=0)

        hist, edges_x, edges_y = np.histogram2d(
            x=position.x,
            y=position.y,
            bins=bins_dict,
            range={'pupil.x': [limit_min.x, limit_max.x], 'pupil.y': [limit_min.y, limit_max.y]},
            weights=mask,
        )

        centers_x = (edges_x[{'pupil.x': slice(1, None)}] + edges_x[{'pupil.x': slice(None, ~0)}]) / 2
        centers_y = (edges_y[{'pupil.y': slice(1, None)}] + edges_y[{'pupil.y': slice(None, ~0)}]) / 2

        return aberrations.PointSpreadFunction(
            input=vectors.SpotVector(
                field=self.input.field,
                position=kgpy.vectors.Cartesian2D(centers_x, centers_y),
                wavelength=self.input.wavelength,
                velocity_los=self.input.velocity_los,
            ),
            output=hist,
        )


    def pupil_hist2d(
            self,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limit_min: typ.Optional['kgpy.vectors.Vector2D'] = None,
            limit_max: typ.Optional['kgpy.vectors.Vector2D'] = None,
            use_vignetted: bool = False,
            relative_to_centroid: typ.Tuple[bool, bool] = (False, False),
            use_position_apparent: bool = False,
    ) -> typ.Tuple[np.ndarray, u.Quantity, u.Quantity]:

        if isinstance(bins, int):
            bins = (bins, bins)

        if not use_vignetted:
            mask = self.mask
        else:
            mask = self.error_mask

        if not use_position_apparent:
            position = self.position.copy()
            position_rel = self.position_relative_pupil
        else:
            position = self.position_apparent
            position_rel = self._calc_relative_pupil(position)

        if relative_to_centroid[kgpy.vectors.ix]:
            position.x = position_rel.x
        if relative_to_centroid[kgpy.vectors.iy]:
            position.y = position_rel.y

        position_masked = position[mask]
        if limit_min is None:
            limit_min = np.nanmin(position_masked).xy
        if limit_max is None:
            limit_max = np.nanmax(position_masked).xy

        limits = np.stack([limit_min.quantity, limit_max.quantity], axis=~0)

        hist_shape = list(self.grid_shape)
        hist_shape[self.axis.pupil_x] = bins[kgpy.vectors.ix]
        hist_shape[self.axis.pupil_y] = bins[kgpy.vectors.iy]
        hist = np.empty(hist_shape)

        edges_x_shape = list(self.grid_shape)
        edges_x_shape[self.axis.pupil_x] = bins[kgpy.vectors.ix] + 1
        edges_x_shape[self.axis.pupil_y] = 1
        edges_x = np.empty(edges_x_shape)

        edges_y_shape = list(self.grid_shape)
        edges_y_shape[self.axis.pupil_x] = 1
        edges_y_shape[self.axis.pupil_y] = bins[kgpy.vectors.iy] + 1
        edges_y = np.empty(edges_y_shape)

        # base_shape = self.shape + self.grid_shape[self.axis.wavelength:self.axis.field_y + 1]
        # hist = np.empty(base_shape + tuple(bins))
        # edges_x = np.empty(base_shape + (bins[vector.ix] + 1,))
        # edges_y = np.empty(base_shape + (bins[vector.iy] + 1,))

        hist_flat = hist.reshape((-1,) + hist.shape[~(self.axis.ndim - 1):])
        edges_x_flat = edges_x.reshape((-1,) + edges_x.shape[~(self.axis.ndim - 1):])
        edges_y_flat = edges_y.reshape((-1,) + edges_y.shape[~(self.axis.ndim - 1):])
        position_flat = position.reshape((-1,) + position.shape[~(self.axis.ndim - 1):])
        mask_flat = mask.reshape((-1,) + mask.shape[~(self.axis.ndim - 1):])

        for c in range(hist_flat.shape[0]):
            for i in range(hist_flat.shape[self.axis.field_x]):
                for j in range(hist_flat.shape[self.axis.field_y]):
                    for w in range(hist_flat.shape[self.axis.wavelength]):
                        for v in range(hist_flat.shape[self.axis.velocity_los]):
                            cijw = [slice(None)] * hist_flat.ndim
                            cijw[0] = c
                            cijw[self.axis.field_x] = i
                            cijw[self.axis.field_y] = j
                            cijw[self.axis.wavelength] = w
                            cijw[self.axis.velocity_los] = v
                            cijwx = cijw.copy()
                            cijwy = cijw.copy()
                            cijwx[self.axis.pupil_y] = 0
                            cijwy[self.axis.pupil_x] = 0
                            hist_flat[cijw], edges_x_flat[cijwx], edges_y_flat[cijwy] = np.histogram2d(
                                x=position_flat[cijw].x.flatten().to(limits.unit).value,
                                y=position_flat[cijw].y.flatten().to(limits.unit).value,
                                bins=bins,
                                weights=mask_flat[cijw].flatten(),
                                range=limits.value,
                            )

        hist = hist_flat.reshape(hist.shape)
        edges_x = edges_x_flat.reshape(edges_x.shape)
        edges_y = edges_y_flat.reshape(edges_y.shape)

        # if not self.shape:
        #     position = position[None, ...]
        #     mask = mask[None, ...]
        #     hist, edges_x, edges_y = hist[None, ...], edges_x[None, ...], edges_y[None, ...]

        # for c, p_c in enumerate(position):
        #     for w, p_cw in enumerate(p_c):
        #         for i, p_cwi in enumerate(p_cw):
        #             for j, p_cwij in enumerate(p_cwi):
        #                 cwij = c, w, i, j
        #                 hist[cwij], edges_x[cwij], edges_y[cwij] = np.histogram2d(
        #                     x=p_cwij.x.flatten().value,
        #                     y=p_cwij.y.flatten().value,
        #                     bins=bins,
        #                     weights=mask[cwij].flatten(),
        #                     range=limits,
        #                 )

        unit = self.position.x.unit
        return hist, edges_x << unit, edges_y << unit

    @classmethod
    def calc_mtf(
            cls,
            psf: u.Quantity,
            limit_min: 'kgpy.vectors.Vector2D',
            limit_max: 'kgpy.vectors.Vector2D',
            # frequency_min: typ.Optional[typ.Union[u.Quantity, kgpy.vector.Vector2D]] = None,
    ) -> typ.Tuple[u.Quantity, 'kgpy.vectors.Vector2D']:

        psf_sum_pupil = np.nansum(a=psf, axis=cls.axis.pupil_xy, keepdims=True)
        psf = np.nan_to_num(psf / psf_sum_pupil)

        print(psf.shape)

        bins = kgpy.vectors.Vector2D.from_tuple(np.array(psf.shape)[np.array(cls.axis.pupil_xy)])

        print('bins', bins)

        period = limit_max - limit_min
        print('period', period)
        # period = (1 * u.dimensionless_unscaled) / frequency_min
        spacing = period / (bins - np.array(1))
        spacing = np.broadcast_to(spacing, psf_sum_pupil.shape, subok=True)
        # print(spacing)

        print(psf.shape)
        mtf = np.abs(np.fft.fft2(a=psf, axes=cls.axis.pupil_xy)) * u.dimensionless_unscaled

        frequency = kgpy.vectors.Vector2D(
            x=np.moveaxis(np.fft.fftfreq(n=bins.x, d=np.moveaxis(spacing.x, cls.axis.pupil_x, ~0)), ~0, cls.axis.pupil_x),
            y=np.moveaxis(np.fft.fftfreq(n=bins.y, d=np.moveaxis(spacing.y, cls.axis.pupil_y, ~0)), ~0, cls.axis.pupil_y),
        )

        print('frequency.shape', frequency.shape)

        # mask_x = frequency.x >= 0
        # mask_y = frequency.y >= 0
        range_x = range(bins.x - bins.x // 2)
        range_y = range(bins.y - bins.y // 2)

        print(range_x)

        mtf = mtf.take(indices=range_x, axis=cls.axis.pupil_x)
        mtf = mtf.take(indices=range_y, axis=cls.axis.pupil_y)

        frequency.x = frequency.x.take(indices=range_x, axis=cls.axis.pupil_x)
        frequency.y = frequency.y.take(indices=range_y, axis=cls.axis.pupil_y)

        print('mtf mean', mtf.mean())

        return mtf, frequency

    def mtf(
            self,
            bins: typ.Union[int, 'kgpy.vectors.Vector2D'] = 10,
            frequency_min: typ.Optional[typ.Union[u.Quantity, 'kgpy.vector.Vector2D']] = None,
            use_vignetted: bool = False,
    ) -> typ.Tuple[u.Quantity, 'kgpy.vectors.Vector2D']:

        if not isinstance(bins, kgpy.vectors.Vector2D):
            bins = kgpy.vectors.Vector2D(x=bins, y=bins)

        if not isinstance(frequency_min, kgpy.vectors.Vector2D):
            frequency_min = kgpy.vectors.Vector2D(x=frequency_min, y=frequency_min)

        period = (1 * u.dimensionless_unscaled) / frequency_min
        limit_max = period / 2
        limit_min = -limit_max


        psf, edges_x, edges_y = self.pupil_hist2d(
            bins=bins.to_tuple(),
            limit_min=limit_min,
            limit_max=limit_max,
            use_vignetted=use_vignetted,
            relative_to_centroid=(True, True),
            use_position_apparent=True,
        )

        return self.calc_mtf(psf=psf, limit_min=limit_min, limit_max=limit_max)



        # psf = np.nan_to_num(psf / np.nansum(a=psf, axis=self.axis.pupil_xy, keepdims=True))
        #
        # print(psf.shape)
        # mtf = np.abs(np.fft.fft2(a=psf, axes=(self.axis.pupil_x, self.axis.pupil_y))) * u.dimensionless_unscaled
        #
        # frequency = kgpy.vector.Vector2D(
        #     x=np.fft.fftfreq(n=bins.x, d=spacing.x),
        #     y=np.fft.fftfreq(n=bins.y, d=spacing.y),
        # )
        #
        # sl = [slice(None)] * mtf.ndim
        # mask_x = frequency.x >= 0
        # mask_y = frequency.y >= 0
        # mesh = np.ix_(mask_x, mask_y)
        # sl[self.axis.pupil_x] = mesh[0]
        # sl[self.axis.pupil_y] = mesh[1]
        # mtf = mtf[sl]
        #
        # frequency.x = frequency.x[mask_x]
        # frequency.y = frequency.y[mask_y]
        #
        # return mtf, frequency


    def colorgrid(self, axis: int) -> np.ndarray:
        grid = self.input_grids[axis]
        return np.broadcast_to(grid, self.shape + grid.shape[~0:], subok=True)

    def colormesh(self, axis: int) -> np.ndarray:
        mesh = np.expand_dims(self.colorgrid(axis), self.axis.perp_axes(axis))
        return np.broadcast_to(mesh, self.grid_shape, subok=True)

    @classmethod
    def calc_labels(cls, name: str, grid: u.Quantity):
        label_func = np.vectorize(lambda g: name + '= {0.value:0.3f} {0.unit:latex}'.format(g << grid.unit))
        return label_func(grid.value)

    def grid_labels(self, axis: int) -> np.ndarray:
        name = self.axis.latex_names[axis]
        grid = self.input_grids[axis]
        return self.calc_labels(name, grid)

    def plot_position(
            self,
            ax: typ.Optional[plt.Axes] = None,
            color_axis: str = 'wavelength',
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        return self.plot_attribute(
            attr_x=self.position.x,
            attr_y=self.position.y,
            ax=ax,
            color_axis=color_axis,
            plot_vignetted=plot_vignetted
        )

    def plot_direction(
            self,
            ax: typ.Optional[plt.Axes] = None,
            color_axis: str = 'wavelength',
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        return self.plot_attribute(
            attr_x=np.arctan(self.direction.x, self.direction.z).to(u.arcmin),
            attr_y=np.arctan(self.direction.y, self.direction.z).to(u.arcmin),
            ax=ax,
            color_axis=color_axis,
            plot_vignetted=plot_vignetted
        )

    def plot_attribute(
            self,
            attr_x: u.Quantity,
            attr_y: u.Quantity,
            ax: typ.Optional[plt.Axes] = None,
            color_axis: str = 'wavelength',
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        attr_x = np.broadcast_to(attr_x, self.grid_shape)
        attr_y = np.broadcast_to(attr_y, self.grid_shape)

        if plot_vignetted:
            mask = self.error_mask
        else:
            mask = self.mask
        mask = np.broadcast_to(mask, self.grid_shape)

        mesh = self.input_grid.points_from_axis(color_axis)
        # sl = self.axis.ndim * [np.newaxis]
        # sl[color_axis] = slice(None)
        # mesh = self.input_grid.grids[color_axis][sl]
        mesh = np.broadcast_to(mesh, self.grid_shape, subok=True)

        with astropy.visualization.quantity_support():
            colormap = plt.cm.viridis
            colornorm = plt.Normalize(vmin=mesh.value.min(), vmax=mesh.value.max())
            color = colormap(colornorm(mesh.value))
            scatter = ax.scatter(
                x=attr_x[mask],
                y=attr_y[mask],
                c=color[mask],
            )
            ax.figure.colorbar(
                plt.cm.ScalarMappable(cmap=colormap, norm=colornorm),
                ax=ax,
                fraction=0.02,
                label=self.axis.latex_names[color_axis] + ' (' + str(mesh.unit) + ')',
            )

            # try:
            #     ax.legend(
            #         handles=scatter.legend_elements(num=self.input_grids[color_axis].flatten())[0],
            #         labels=list(self.grid_labels(color_axis).flatten()),
            #         loc='center left',
            #         bbox_to_anchor=(1.0, 0.5),
            #     )
            # except ValueError:
            #     pass

        return ax

    def plot_pupil_hist2d_vs_field(
            self,
            config_index: int = 0,
            wavlen_index: int = 0,
            velocity_los_index: int = 0,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limit_min: typ.Optional['kgpy.vectors.Vector2D'] = None,
            limit_max: typ.Optional['kgpy.vectors.Vector2D'] = None,
            use_vignetted: bool = False,
            relative_to_centroid: typ.Tuple[bool, bool] = (True, True),
            norm: typ.Optional[matplotlib.colors.Normalize] = None,
            cmap: str = 'viridis',
            kwargs_colorbar: typ.Optional[typ.Dict[str, typ.Any]] = None,
    ) -> typ.Tuple[plt.Figure, np.ndarray]:

        if kwargs_colorbar is None:
            kwargs_colorbar = {}

        # field_x = self.input_grids[self.axis.field_x]
        # field_y = self.input_grids[self.axis.field_y]

        field_x = self.input_grid.field.points.x
        field_y = self.input_grid.field.points.y

        hist, edges_x, edges_y = self.pupil_hist2d(
            bins=bins,
            limit_min=limit_min,
            limit_max=limit_max,
            use_vignetted=use_vignetted,
            relative_to_centroid=relative_to_centroid,
        )

        fig, axs = plt.subplots(
            nrows=self.grid_shape[self.axis.field_x],
            ncols=self.grid_shape[self.axis.field_y],
            sharex='all',
            sharey='all',
            squeeze=False,
            constrained_layout=True,
        )

        if hist.ndim > self.axis.ndim:
            hist, edges_x, edges_y = hist[config_index], edges_x[config_index], edges_y[config_index]

        for i, axs_i in enumerate(reversed(axs)):
            for j, axs_ij in enumerate(axs_i):
                axs_ij.invert_xaxis()
                cwji = [slice(None)] * hist.ndim
                cwji[self.axis.wavelength] = wavlen_index
                cwji[self.axis.velocity_los] = velocity_los_index
                cwji[self.axis.field_x] = j
                cwji[self.axis.field_y] = i
                if hist[cwji].sum() > 0:
                    w = [slice(None)] * hist.ndim
                    w[self.axis.wavelength] = wavlen_index
                    limits = [
                        edges_x[cwji].min().value,
                        edges_x[cwji].max().value,
                        edges_y[cwji].min().value,
                        edges_y[cwji].max().value,
                    ]
                    img = axs_ij.imshow(
                        X=hist[cwji].T,
                        extent=limits,
                        aspect='equal',
                        origin='lower',
                        vmin=hist[w].min(),
                        vmax=hist[w].max(),
                        norm=norm,
                        cmap=cmap,
                    )
                else:
                    axs_ij.spines['top'].set_visible(False)
                    axs_ij.spines['right'].set_visible(False)
                    axs_ij.spines['bottom'].set_visible(False)
                    axs_ij.spines['left'].set_visible(False)

                if i == len(axs) - 1:
                    axs_ij.set_xlabel(kgpy.format.quantity(field_x[j], digits_after_decimal=1))
                    axs_ij.xaxis.set_label_position('top')
                elif i == 0:
                    axs_ij.set_xlabel(edges_x.unit)

                if j == 0:
                    axs_ij.set_ylabel(edges_y.unit)
                elif j == len(axs_i) - 1:
                    axs_ij.yaxis.set_label_position('right')
                    axs_ij.set_ylabel(
                        kgpy.format.quantity(field_y[i], digits_after_decimal=1),
                        rotation='horizontal',
                        ha='left',
                        va='center',
                    )

                # axs_ij.tick_params(axis='both', labelsize=8)

        # wavelength = self.input_grids[self.axis.wavelength]
        wavelength = self.input_grid.wavelength.points
        if wavelength.ndim > 1:
            wavelength = wavelength[config_index]
        wavl_str = wavelength[wavlen_index]
        wavl_str = '{0.value:0.3f} {0.unit:latex}'.format(wavl_str)
        fig.suptitle('configuration = ' + str(config_index) + ', wavelength = ' + wavl_str)
        fig.colorbar(img, ax=axs, fraction=0.05, **kwargs_colorbar)

        return fig, axs


@dataclasses.dataclass
class RayFunctionList(
    kgpy.io.dxf.WritableMixin,
    kgpy.mixin.Plottable,
    kgpy.mixin.DataclassList[RayFunction],
):
    @property
    def intercepts(self) -> kgpy.vectors.Cartesian3D:
        intercepts = []
        for rays in self:
            intercept = rays.output.transform(rays.output.position)
            intercept = np.broadcast_to(intercept, self[~0].shape)
            intercepts.append(intercept)
        return np.stack(intercepts, axis='surface')

    def plot(
            self,
            ax: plt.Axes,
            component_x: str = 'x',
            component_y: str = 'y',
            component_z: str = 'z',
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            color_axis: str = 'wavelength',
            plot_vignetted: bool = False,
            colormap: typ.Optional[matplotlib.cm.ScalarMappable] = None,
            # plot_colorbar: bool = True,
            **kwargs,
    ) -> typ.Tuple[typ.List[plt.Line2D], typ.Optional[matplotlib.colorbar.Colorbar]]:

        kwargs = {**self.plot_kwargs, **kwargs}

        if transform_extra is None:
            transform_extra = kgpy.transforms.TransformList()

        img_rays = self[~0]

        intercepts = transform_extra(self.intercepts)

        # color_axis = (color_axis % img_rays.axis.ndim) - img_rays.axis.ndim

        if plot_vignetted:
            mask = None
        else:
            mask = img_rays.output.mask

        color_axis = color_axis.split('.')
        color = img_rays.input
        for element in color_axis:
            color = color.coordinates[element]

        return intercepts.plot(
            ax=ax,
            axis_plot='surface',
            where=mask,
            color=color,
            colormap=colormap,
            component_x=component_x,
            component_y=component_y,
            component_z=component_z,
            **kwargs
        )

        # if isinstance(coordinates_color, kgpy.uncertainty.AbstractArray):
        #     coordinates_color = coordinates_color.distribution

        # colormap = matplotlib.cm.ScalarMappable(
        #     cmap=plt.cm.viridis,
        #     norm=plt.Normalize(
        #         vmin=coordinates_color.min().array,
        #         vmax=coordinates_color.max().array,
        #     )
        # )
        #
        # mesh = img_rays.input_grid.points_from_axis(color_axis)
        # mesh = np.broadcast_to(mesh, img_rays.grid_shape, subok=True)
        #
        # with astropy.visualization.quantity_support():
        #     colormap = plt.cm.viridis
        #     colornorm = plt.Normalize(vmin=mesh.value.min(), vmax=mesh.value.max())
        #     if mesh.value.min() == mesh.value.max():
        #         color = np.broadcast_to(colormap(0.5), mesh.shape + (4, ))
        #     else:
        #         color = colormap(colornorm(mesh.value))
        #
        #     intercepts = intercepts[:, mask]
        #     color = color[mask]
        #
        #     lines = []
        #     for i in range(intercepts.shape[~0]):
        #         plot_kwargs_z = {}
        #         if component_z is not None:
        #             plot_kwargs_z['zs'] = intercepts[..., i].get_component(component_z)
        #         if 'color' not in plot_kwargs:
        #             kwargs_color = dict(color=color[..., i, :])
        #         else:
        #             kwargs_color = dict()
        #         lines_i = ax.plot(
        #             intercepts[..., i].get_component(components[0]),
        #             intercepts[..., i].get_component(components[1]),
        #             **kwargs_color,
        #             **plot_kwargs_z,
        #             **plot_kwargs,
        #         )
        #
        #         lines = lines + lines_i
        #
        #     if plot_colorbar:
        #         colorbar = ax.figure.colorbar(
        #             matplotlib.cm.ScalarMappable(cmap=colormap, norm=colornorm),
        #             ax=ax, fraction=0.02,
        #             label=img_rays.axis.latex_names[color_axis] + ' (' + str(mesh.unit) + ')',
        #         )
        #     else:
        #         colorbar = None
        #
        # return lines, colorbar

    def to_dxf(self, filename: pathlib.Path, dxf_unit: u.Unit = u.imperial.inch):

        import ezdxf.addons
        with ezdxf.addons.r12writer(filename) as dxf:

            mask = np.broadcast_to(self[~1].mask, self[~0].mask.shape)

            intercepts = self.intercepts[:, mask]

            intercepts = kgpy.vectors.Vector3D(x=-intercepts.z, y=-intercepts.x, z=intercepts.y)

            # intercepts = intercepts.reshape(intercepts.shape[0], -1)
            axis = 1
            for i in range(intercepts.shape[axis]):
                dxf.add_polyline(intercepts.take(indices=i, axis=axis).quantity.to(dxf_unit).value)

    def write_to_dxf(
            self: RaysListT,
            file_writer: R12FastStreamWriter,
            unit: u.Unit,
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
    ) -> None:

        mask = np.broadcast_to(self[~1].mask, self[~0].mask.shape)

        intercepts = self.intercepts[:, mask]

        if transform_extra is not None:
            intercepts = transform_extra(intercepts)

        axis = 1
        for i in range(intercepts.shape[axis]):
            file_writer.add_polyline(intercepts.take(indices=i, axis=axis).quantity.to(unit).value)

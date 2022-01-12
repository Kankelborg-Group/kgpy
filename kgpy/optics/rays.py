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
from kgpy import mixin, vector, transform, format as fmt, grid
import kgpy.dxf
from .aberration import Distortion, Vignetting, Aberration

__all__ = [
    'Axis',
    'RayGrid',
    'Rays',
    'RaysList',
]

RaysListT = typ.TypeVar('RaysListT', bound='RaysList')


class Axis(mixin.AutoAxis):
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


@dataclasses.dataclass
class RayGrid(
    mixin.Copyable,
    abc.ABC,
):
    axis: typ.ClassVar[Axis] = Axis()
    field: grid.RegularGrid2D = dataclasses.field(default_factory=grid.RegularGrid2D)
    pupil: grid.RegularGrid2D = dataclasses.field(default_factory=grid.RegularGrid2D)
    wavelength: grid.Grid1D = dataclasses.field(default_factory=lambda: grid.RegularGrid1D(min=0 * u.nm, max=0 * u.nm))
    velocity_los: grid.Grid1D = dataclasses.field(
        default_factory=lambda: grid.RegularGrid1D(min=0 * u.km / u.s, max=0 * u.km / u.s)
    )

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return np.broadcast(
            np.expand_dims(self.field.points.x, self.axis.perp_axes(self.axis.field_x)),
            np.expand_dims(self.field.points.y, self.axis.perp_axes(self.axis.field_y)),
            np.expand_dims(self.pupil.points.x, self.axis.perp_axes(self.axis.pupil_x)),
            np.expand_dims(self.pupil.points.y, self.axis.perp_axes(self.axis.pupil_y)),
            np.expand_dims(self.wavelength.points, self.axis.perp_axes(self.axis.wavelength)),
            np.expand_dims(self.velocity_los.points, self.axis.perp_axes(self.axis.velocity_los)),
        ).shape

    @property
    def points_field(self) -> vector.Vector2D:
        return self.field.mesh(shape=self.shape, new_axes=self.axis.perp_axes([self.axis.field_x, self.axis.field_y]))

    @property
    def points_pupil(self) -> vector.Vector2D:
        return self.pupil.mesh(shape=self.shape, new_axes=self.axis.perp_axes([self.axis.pupil_x, self.axis.pupil_y]))

    @property
    def points_wavelength(self) -> u.Quantity:
        return self.wavelength.mesh(shape=self.shape, new_axes=self.axis.perp_axes(self.axis.wavelength))

    @property
    def points_velocity_los(self) -> u.Quantity:
        return self.velocity_los.mesh(shape=self.shape, new_axes=self.axis.perp_axes(self.axis.velocity_los))

    def points(self, component_axis: int = ~0) -> u.Quantity:

        points_field = self.points_field
        points_pupil = self.points_pupil

        p = [None] * self.axis.ndim
        p[self.axis.field_x] = points_field.x
        p[self.axis.field_y] = points_field.y
        p[self.axis.pupil_x] = points_pupil.x
        p[self.axis.pupil_y] = points_pupil.y
        p[self.axis.wavelength] = self.points_wavelength
        p[self.axis.velocity_los] = self.points_velocity_los
        return np.stack(arrays=p, axis=component_axis)

    # @property
    # def grids(self) -> typ.List[u.Quantity]:
    #     return [
    #         self.field.points.x,
    #         self.field.points.y,
    #         self.pupil.points.x,
    #         self.pupil.points.y,
    #         self.wavelength.points,
    #         self.velocity_los.points,
    #     ]

    def points_from_axis(self, axis: int):
        if axis == self.axis.field_x:
            return self.points_field.x
        elif axis == self.axis.field_y:
            return self.points_field.y
        elif axis == self.axis.pupil_x:
            return self.points_pupil.x
        elif axis == self.axis.pupil_y:
            return self.points_pupil.y
        elif axis == self.axis.wavelength:
            return self.points_wavelength
        elif axis == self.axis.velocity_los:
            return self.points_velocity_los
        else:
            raise ValueError('Unsupported axis')


@dataclasses.dataclass
class Rays(transform.rigid.Transformable):
    axis = Axis()

    # wavelength: u.Quantity = dataclasses.field(default_factory=lambda: [[0]] * u.nm)
    # position: u.Quantity = dataclasses.field(default_factory=lambda: [[0, 0, 0]] * u.mm)
    # direction: u.Quantity = dataclasses.field(default_factory=lambda: [[0, 0, 1]] * u.dimensionless_unscaled)
    # polarization: u.Quantity = dataclasses.field(default_factory=lambda: [[1, 0]] * u.dimensionless_unscaled)
    # surface_normal: u.Quantity = dataclasses.field(default_factory=lambda: [[0, 0, -1]] * u.dimensionless_unscaled)
    # index_of_refraction: u.Quantity = dataclasses.field(default_factory=lambda: [[1]] * u.dimensionless_unscaled)
    intensity: u.Quantity = 1 * u.dimensionless_unscaled
    wavelength: u.Quantity = 0 * u.nm
    position: vector.Vector3D = dataclasses.field(default_factory=vector.Vector3D)
    direction: vector.Vector3D = dataclasses.field(default_factory=vector.zhat_factory)
    velocity_los: u.Quantity = 0 * u.km / u.s
    surface_normal: vector.Vector3D = dataclasses.field(default_factory=lambda: -vector.zhat_factory())
    index_of_refraction: u.Quantity = 1 * u.dimensionless_unscaled
    vignetted_mask: np.ndarray = np.array([True])
    error_mask: np.ndarray = np.array([True])
    input_grid: typ.Optional[RayGrid] = None
    distortion_polynomial_degree: int = 2
    vignetting_polynomial_degree: int = 1

    # input_wavelength: typ.Optional[u.Quantity] = None
    # input_field: typ.Optional[vector.Vector2D] = None
    # input_pupil: typ.Optional[vector.Vector2D] = None
    # input_velocity_z: typ.Optional[u.Quantity] = None

    # input_grids: typ.List[typ.Optional[u.Quantity]] = dataclasses.field(
    #     default_factory=lambda: [None, None, None, None, None],
    # )

    @property
    def field_angles(self) -> vector.Vector2D:
        angle = np.arcsin(self.direction.xy).to(u.deg)
        angle.y = -angle.y
        return angle

    @classmethod
    def from_field_angles(
            cls,
            # wavelength_grid: u.Quantity,
            input_grid: RayGrid,
            position: vector.Vector3D,
            # field_grid: vector.Vector2D,
            # pupil_grid: vector.Vector2D,
            # velocity_z_grid: u.Quantity
    ) -> 'Rays':

        # field_x = np.expand_dims(input_grid.field.points.x, cls.axis.perp_axes(cls.axis.field_x))
        # field_y = np.expand_dims(input_grid.field.points.y, cls.axis.perp_axes(cls.axis.field_y))

        direction = transform.rigid.TiltX(input_grid.points_field.y)(vector.z_hat)
        direction = transform.rigid.TiltY(input_grid.points_field.x)(direction)

        return cls(
            wavelength=input_grid.points_wavelength,
            position=position,
            direction=direction,
            velocity_los=input_grid.points_velocity_los,
            input_grid=input_grid,
            # input_wavelength=wavelength_grid,
            # input_field=field_grid,
            # input_pupil=pupil_grid,
            # input_velocity_z=velocity_z_grid,
        )

    @classmethod
    def from_field_positions(
            cls,
            # intensity: u.Quantity,
            # wavelength_grid: u.Quantity,
            input_grid: RayGrid,
            direction: vector.Vector3D,
            # field_grid: vector.Vector2D,
            # pupil_grid: vector.Vector2D,
            # velocity_z_grid: u.Quantity,
    ) -> 'Rays':

        return cls(
            wavelength=input_grid.points_wavelength,
            position=input_grid.points_field.to_3d(z=0 * u.mm),
            direction=direction,
            velocity_los=input_grid.points_velocity_los,
            input_grid=input_grid,
            # input_wavelength=wavelength_grid,
            # input_field=field_grid,
            # input_pupil=pupil_grid,
            # input_velocity_z=velocity_z_grid,
        )

    def apply_transform_list(self, transform_list: transform.rigid.TransformList) -> 'Rays':
        # other = self.copy()
        other = self.copy_shallow()
        transform_list = transform_list.simplified
        other.position = transform_list(other.position, num_extra_dims=self.axis.ndim)
        other.direction = transform_list(other.direction, translate=False, num_extra_dims=self.axis.ndim)
        other.surface_normal = transform_list(other.surface_normal, translate=False, num_extra_dims=self.axis.ndim)
        other.transform = self.transform + transform_list.inverse
        return other

    @property
    def transformed(self) -> 'Rays':
        other = self.apply_transform_list(self.transform)
        other.transform = transform.rigid.TransformList()
        return other

    @property
    def grid_shape(self) -> typ.Tuple[int, ...]:
        return np.broadcast(
            self.wavelength,
            self.position,
            self.direction,
            self.mask,
        ).shape

    # @property
    # def vector_grid_shape(self) -> typ.Tuple[int, ...]:
    #     return self.grid_shape + (3,)
    #
    # @property
    # def scalar_grid_shape(self) -> typ.Tuple[int, ...]:
    #     return self.grid_shape + (1,)

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.grid_shape[:~(self.axis.ndim - 1)]

    @property
    def base_shape(self):
        return self.grid_shape[~(self.axis.ndim - 1)]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(np.array(self.shape)))

    @property
    def num_wavlength(self):
        return self.grid_shape[self.axis.wavelength]

    # def _input_grid(self, axis: int) -> u.Quantity:
    #     grid = self.input_grids[axis]
    #     return np.broadcast_to(grid, self.shape + grid.shape[~0:], subok=True)

    @property
    def mask(self) -> np.ndarray:
        return self.vignetted_mask & self.error_mask

    # @property
    # def input_field_mesh(self) -> vector.Vector2D:
    #     return vector.Vector2D(
    #         x=self.input_field.x[..., :, np.newaxis],
    #         y=self.input_field.y[..., np.newaxis, :],
    #     )

    @property
    def energy(self) -> u.Quantity:
        return (astropy.constants.h * astropy.constants.c / self.wavelength).to(u.eV)

    def _calc_avg_pupil(self, a: vector.Vector3D) -> vector.Vector3D:
        a = a.copy()
        a[~self.mask] = np.nan
        return np.nanmean(a=a, axis=self.axis.pupil_xy, keepdims=True)

    def _calc_relative_pupil(self, a: vector.Vector3D) -> vector.Vector3D:
        return a - self._calc_avg_pupil(a)

    @property
    def position_avg_pupil(self) -> vector.Vector3D:
        return self._calc_avg_pupil(self.position)

    @property
    def position_relative_pupil(self) -> vector.Vector3D:
        return self._calc_relative_pupil(self.position)

    @property
    def position_apparent(self):
        position = self.position.copy()
        position.z = np.broadcast_to(self.wavelength, position.shape, subok=True).copy()
        position = self.distortion.model(inverse=True)(position)
        return position

    @property
    def distortion(self) -> Distortion:
        return Distortion(
            wavelength=self.input_grid.wavelength.points[..., np.newaxis, np.newaxis, :],
            spatial_mesh_input=vector.Vector2D(
                x=self.input_grid.field.points.x[..., :, np.newaxis, np.newaxis],
                y=self.input_grid.field.points.y[..., np.newaxis, :, np.newaxis],
            ),
            spatial_mesh_output=self.position_avg_pupil[..., 0, 0, :, 0].xy,
            mask=self.mask.any((self.axis.pupil_x, self.axis.pupil_y, self.axis.velocity_los)),
            polynomial_degree=self.distortion_polynomial_degree,
        )

    @property
    def vignetting(self) -> Vignetting:
        intensity = self.intensity.copy()
        intensity, mask = np.broadcast_arrays(intensity, self.mask, subok=True)
        intensity[~mask] = 0
        counts = intensity.sum((self.axis.pupil_x, self.axis.pupil_y, self.axis.velocity_los))
        return Vignetting(
            wavelength=self.input_grid.wavelength.points[..., np.newaxis, np.newaxis, :],
            spatial_mesh=vector.Vector2D(
                x=self.input_grid.field.points.x[..., :, np.newaxis, np.newaxis],
                y=self.input_grid.field.points.y[..., np.newaxis, :, np.newaxis],
            ),
            unvignetted_percent=counts,
            mask=mask.any((self.axis.pupil_x, self.axis.pupil_y, self.axis.velocity_los)),
            polynomial_degree=self.vignetting_polynomial_degree,
        )

    @property
    def aberration(self) -> Aberration:
        return Aberration(
            distortion=self.distortion,
            vignetting=self.vignetting,
        )

    @property
    def spot_size_rms(self):
        position = self.position_relative_pupil
        r = position.xy.length
        r2 = np.square(r)
        pupil_axes = self.axis.pupil_x, self.axis.pupil_y
        sz = np.sqrt(np.ma.average(r2.value, axis=pupil_axes, weights=self.mask) << r2.unit)
        mask = self.mask.any(pupil_axes)
        sz[~mask] = np.nan
        return sz

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
            wavelength_formatted = fmt.quantity(wavelength[i], digits_after_decimal=digits_after_decimal)
            axs[i].set_title(f'{wavelength_name[i]} {wavelength_formatted}')
            sl = [slice(None)] * sizes.ndim
            sl[self.axis.wavelength] = i
            sl[self.axis.velocity_los] = velocity_los_index

            extent = kgpy.plot.calc_extent(
                data_min=vector.Vector2D(field_x.min(), field_y.min()),
                data_max=vector.Vector2D(field_x.max(), field_y.max()),
                num_steps=vector.Vector2D.from_quantity(sizes[sl].shape * u.dimensionless_unscaled),
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

    def pupil_hist2d(
            self,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limit_min: typ.Optional[kgpy.vector.Vector2D] = None,
            limit_max: typ.Optional[kgpy.vector.Vector2D] = None,
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

        if relative_to_centroid[vector.ix]:
            position.x = position_rel.x
        if relative_to_centroid[vector.iy]:
            position.y = position_rel.y

        position_masked = position[mask]
        if limit_min is None:
            limit_min = np.nanmin(position_masked).xy
        if limit_max is None:
            limit_max = np.nanmax(position_masked).xy

        limits = np.stack([limit_min.quantity, limit_max.quantity], axis=~0)

        hist_shape = list(self.grid_shape)
        hist_shape[self.axis.pupil_x] = bins[vector.ix]
        hist_shape[self.axis.pupil_y] = bins[vector.iy]
        hist = np.empty(hist_shape)

        edges_x_shape = list(self.grid_shape)
        edges_x_shape[self.axis.pupil_x] = bins[vector.ix] + 1
        edges_x_shape[self.axis.pupil_y] = 1
        edges_x = np.empty(edges_x_shape)

        edges_y_shape = list(self.grid_shape)
        edges_y_shape[self.axis.pupil_x] = 1
        edges_y_shape[self.axis.pupil_y] = bins[vector.iy] + 1
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
            limit_min: kgpy.vector.Vector2D,
            limit_max: kgpy.vector.Vector2D,
            # frequency_min: typ.Optional[typ.Union[u.Quantity, kgpy.vector.Vector2D]] = None,
    ) -> typ.Tuple[u.Quantity, kgpy.vector.Vector2D]:

        psf_sum_pupil = np.nansum(a=psf, axis=cls.axis.pupil_xy, keepdims=True)
        psf = np.nan_to_num(psf / psf_sum_pupil)

        print(psf.shape)

        bins = kgpy.vector.Vector2D.from_tuple(np.array(psf.shape)[np.array(cls.axis.pupil_xy)])

        print('bins', bins)

        period = limit_max - limit_min
        print('period', period)
        # period = (1 * u.dimensionless_unscaled) / frequency_min
        spacing = period / (bins - np.array(1))
        spacing = np.broadcast_to(spacing, psf_sum_pupil.shape, subok=True)
        # print(spacing)

        print(psf.shape)
        mtf = np.abs(np.fft.fft2(a=psf, axes=cls.axis.pupil_xy)) * u.dimensionless_unscaled

        frequency = kgpy.vector.Vector2D(
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
            bins: typ.Union[int, kgpy.vector.Vector2D] = 10,
            frequency_min: typ.Optional[typ.Union[u.Quantity, kgpy.vector.Vector2D]] = None,
            use_vignetted: bool = False,
    ) -> typ.Tuple[u.Quantity, kgpy.vector.Vector2D]:

        if not isinstance(bins, kgpy.vector.Vector2D):
            bins = kgpy.vector.Vector2D(x=bins, y=bins)

        if not isinstance(frequency_min, kgpy.vector.Vector2D):
            frequency_min = kgpy.vector.Vector2D(x=frequency_min, y=frequency_min)

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
            color_axis: int = axis.wavelength,
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
            color_axis: int = axis.wavelength,
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
            color_axis: int = axis.wavelength,
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
            limit_min: typ.Optional[kgpy.vector.Vector2D] = None,
            limit_max: typ.Optional[kgpy.vector.Vector2D] = None,
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
                    axs_ij.set_xlabel(fmt.quantity(field_x[j], digits_after_decimal=1))
                    axs_ij.xaxis.set_label_position('top')
                elif i == 0:
                    axs_ij.set_xlabel(edges_x.unit)

                if j == 0:
                    axs_ij.set_ylabel(edges_y.unit)
                elif j == len(axs_i) - 1:
                    axs_ij.yaxis.set_label_position('right')
                    axs_ij.set_ylabel(
                        fmt.quantity(field_y[i], digits_after_decimal=1),
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
class RaysList(
    kgpy.dxf.WritableMixin,
    mixin.Plottable,
    mixin.DataclassList[Rays],
):
    @property
    def intercepts(self) -> vector.Vector3D:
        intercepts = []
        for rays in self:
            intercept = rays.transform(rays.position, num_extra_dims=rays.axis.ndim)
            intercept = np.broadcast_to(intercept, self[~0].grid_shape, subok=True)
            intercepts.append(intercept)
        return np.stack(intercepts)

    def plot(
            self,
            ax: plt.Axes,
            components: typ.Tuple[str, str] = ('x', 'y'),
            component_z: typ.Optional[str] = None,
            plot_kwargs: typ.Optional[typ.Dict[str, typ.Any]] = None,
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            color_axis: int = Rays.axis.wavelength,
            plot_vignetted: bool = False,
            plot_colorbar: bool = True,
    ) -> typ.Tuple[typ.List[plt.Line2D], typ.Optional[matplotlib.colorbar.Colorbar]]:

        if plot_kwargs is not None:
            plot_kwargs = {**self.plot_kwargs, **plot_kwargs}
        else:
            plot_kwargs = self.plot_kwargs

        if transform_extra is None:
            transform_extra = transform.rigid.TransformList()

        img_rays = self[~0]

        intercepts = transform_extra(self.intercepts)

        color_axis = (color_axis % img_rays.axis.ndim) - img_rays.axis.ndim

        if plot_vignetted:
            mask = img_rays.error_mask
        else:
            mask = img_rays.mask
        mask = np.broadcast_to(mask, img_rays.grid_shape)

        mesh = img_rays.input_grid.points_from_axis(color_axis)
        mesh = np.broadcast_to(mesh, img_rays.grid_shape, subok=True)

        with astropy.visualization.quantity_support():
            colormap = plt.cm.viridis
            colornorm = plt.Normalize(vmin=mesh.value.min(), vmax=mesh.value.max())
            if mesh.value.min() == mesh.value.max():
                color = np.broadcast_to(colormap(0.5), mesh.shape + (4, ))
            else:
                color = colormap(colornorm(mesh.value))

            intercepts = intercepts[:, mask]
            color = color[mask]

            lines = []
            for i in range(intercepts.shape[~0]):
                plot_kwargs_z = {}
                if component_z is not None:
                    plot_kwargs_z['zs'] = intercepts[..., i].get_component(component_z)
                if 'color' not in plot_kwargs:
                    kwargs_color = dict(color=color[..., i, :])
                else:
                    kwargs_color = dict()
                lines_i = ax.plot(
                    intercepts[..., i].get_component(components[0]),
                    intercepts[..., i].get_component(components[1]),
                    **kwargs_color,
                    **plot_kwargs_z,
                    **plot_kwargs,
                )

                lines = lines + lines_i

            if plot_colorbar:
                colorbar = ax.figure.colorbar(
                    matplotlib.cm.ScalarMappable(cmap=colormap, norm=colornorm),
                    ax=ax, fraction=0.02,
                    label=img_rays.axis.latex_names[color_axis] + ' (' + str(mesh.unit) + ')',
                )
            else:
                colorbar = None

        return lines, colorbar

    def to_dxf(self, filename: pathlib.Path, dxf_unit: u.Unit = u.imperial.inch):

        import ezdxf.addons
        with ezdxf.addons.r12writer(filename) as dxf:

            mask = np.broadcast_to(self[~1].mask, self[~0].mask.shape)

            intercepts = self.intercepts[:, mask]

            intercepts = kgpy.vector.Vector3D(x=-intercepts.z, y=-intercepts.x, z=intercepts.y)

            # intercepts = intercepts.reshape(intercepts.shape[0], -1)
            axis = 1
            for i in range(intercepts.shape[axis]):
                dxf.add_polyline(intercepts.take(indices=i, axis=axis).quantity.to(dxf_unit).value)

    def write_to_dxf(
            self: RaysListT,
            file_writer: R12FastStreamWriter,
            unit: u.Unit,
            transform_extra: typ.Optional[kgpy.transform.rigid.Transform] = None,
    ) -> None:

        mask = np.broadcast_to(self[~1].mask, self[~0].mask.shape)

        intercepts = self.intercepts[:, mask]

        if transform_extra is not None:
            intercepts = transform_extra(intercepts)

        axis = 1
        for i in range(intercepts.shape[axis]):
            file_writer.add_polyline(intercepts.take(indices=i, axis=axis).quantity.to(unit).value)

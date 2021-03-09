import dataclasses
import collections
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import astropy.units as u
import astropy.visualization
import astropy.modeling
# import kgpy.transform.rigid.transform_list
from kgpy import mixin, vector, transform, format as fmt
from .aberration import Distortion, Vignetting, Aberration

__all__ = ['Rays', 'RaysList']


class CAxis(mixin.AutoAxis):
    def __init__(self):
        super().__init__()
        self.components = self.auto_axis_index()


class Axis(mixin.AutoAxis):

    ndim_pupil: typ.ClassVar[int] = 2
    ndim_field: typ.ClassVar[int] = 2

    def __init__(self):
        super().__init__()
        self.wavelength = self.auto_axis_index()
        self.pupil_y = self.auto_axis_index()
        self.pupil_x = self.auto_axis_index()
        self.field_y = self.auto_axis_index()
        self.field_x = self.auto_axis_index()
        # self.wavelength = self.auto_axis_index()

    @property
    def latex_names(self) -> typ.List[str]:
        return [
            # '$\\lambda$',
            '$f_x$', '$f_y$',
            '$p_x$', '$p_y$',
            '$\\lambda$',
        ]


class VAxis(Axis, CAxis):
    pass


@dataclasses.dataclass
class Rays(transform.rigid.Transformable):
    axis = Axis()

    # wavelength: u.Quantity = dataclasses.field(default_factory=lambda: [[0]] * u.nm)
    # position: u.Quantity = dataclasses.field(default_factory=lambda: [[0, 0, 0]] * u.mm)
    # direction: u.Quantity = dataclasses.field(default_factory=lambda: [[0, 0, 1]] * u.dimensionless_unscaled)
    # polarization: u.Quantity = dataclasses.field(default_factory=lambda: [[1, 0]] * u.dimensionless_unscaled)
    # surface_normal: u.Quantity = dataclasses.field(default_factory=lambda: [[0, 0, -1]] * u.dimensionless_unscaled)
    # index_of_refraction: u.Quantity = dataclasses.field(default_factory=lambda: [[1]] * u.dimensionless_unscaled)
    wavelength: u.Quantity = 0 * u.nm
    position: vector.Vector3D = dataclasses.field(default_factory=vector.Vector3D)
    direction: vector.Vector3D = dataclasses.field(default_factory=vector.zhat_factory)
    surface_normal: vector.Vector3D = dataclasses.field(default_factory=lambda: -vector.zhat_factory())
    index_of_refraction: u.Quantity = 1 * u.dimensionless_unscaled
    vignetted_mask: np.ndarray = np.array([True])
    error_mask: np.ndarray = np.array([True])
    input_grids: typ.List[typ.Optional[u.Quantity]] = dataclasses.field(
        default_factory=lambda: [None, None, None, None, None],
    )

    @property
    def field_angles(self) -> vector.Vector2D:
        return -np.arcsin(self.direction.xy).to(u.deg)

    @classmethod
    def from_field_angles(
            cls,
            wavelength_grid: u.Quantity,
            position: vector.Vector3D,
            field_grid_x: u.Quantity,
            field_grid_y: u.Quantity,
            pupil_grid_x: typ.Optional[u.Quantity] = None,
            pupil_grid_y: typ.Optional[u.Quantity] = None,
    ) -> 'Rays':

        wavelength = np.expand_dims(wavelength_grid, cls.axis.perp_axes(cls.axis.wavelength))
        field_x = np.expand_dims(field_grid_x, cls.axis.perp_axes(cls.axis.field_x))
        field_y = np.expand_dims(field_grid_y, cls.axis.perp_axes(cls.axis.field_y))
        # wavelength, field_x, field_y = np.broadcast_arrays(wavelength, field_x, field_y, subok=True)

        # position = np.broadcast_to(position, wavelength.shape)
        # position, _ = np.broadcast_arrays(position, wavelength, subok=True)

        # direction = np.zeros(position.shape)
        # direction[z] = 1
        direction = transform.rigid.TiltX(field_y)(vector.z_hat)
        direction = transform.rigid.TiltY(field_x)(direction)

        input_grids = [None] * cls.axis.ndim
        input_grids[cls.axis.field_x] = field_grid_x
        input_grids[cls.axis.field_y] = field_grid_y
        input_grids[cls.axis.pupil_x] = pupil_grid_x
        input_grids[cls.axis.pupil_y] = pupil_grid_y
        input_grids[cls.axis.wavelength] = wavelength_grid

        return cls(
            wavelength=wavelength,
            position=position,
            direction=direction,
            input_grids=input_grids
        )

    @classmethod
    def from_field_positions(
            cls,
            wavelength_grid: u.Quantity,
            direction: vector.Vector3D,
            field_grid_x: u.Quantity,
            field_grid_y: u.Quantity,
            pupil_grid_x: typ.Optional[u.Quantity] = None,
            pupil_grid_y: typ.Optional[u.Quantity] = None,
    ):
        wavelength = np.expand_dims(wavelength_grid, cls.axis.perp_axes(cls.axis.wavelength))
        # pupil_x = np.expand_dims(pupil_grid_x, cls.axis.perp_axes(cls.axis.pupil_x))
        # pupil_y = np.expand_dims(pupil_grid_y, cls.axis.perp_axes(cls.axis.pupil_y))
        # wavelength, pupil_x, pupil_y = np.broadcast_arrays(wavelength, pupil_x, pupil_y, subok=True)

        # direction, _ = np.broadcast_arrays(direction, wavelength, subok=True)

        position = vector.Vector3D(
            x=np.expand_dims(field_grid_x, cls.axis.perp_axes(cls.axis.field_x)),
            y=np.expand_dims(field_grid_y, cls.axis.perp_axes(cls.axis.field_y)),
            z=0 * u.mm,
        )
        # position, _ = np.broadcast_arrays(position, wavelength, subok=True)

        input_grids = [None] * cls.axis.ndim
        input_grids[cls.axis.field_x] = field_grid_x
        input_grids[cls.axis.field_y] = field_grid_y
        input_grids[cls.axis.pupil_x] = pupil_grid_x
        input_grids[cls.axis.pupil_y] = pupil_grid_y
        input_grids[cls.axis.wavelength] = wavelength_grid

        return cls(
            wavelength=wavelength,
            position=position,
            direction=direction,
            input_grids=input_grids,
        )

    def apply_transform_list(self, transform_list: transform.rigid.TransformList) -> 'Rays':
        other = self.copy()
        other.position = transform_list(other.position, num_extra_dims=self.axis.ndim)
        other.direction = transform_list(other.direction, translate=False, num_extra_dims=self.axis.ndim)
        other.surface_normal = transform_list(other.surface_normal, translate=False, num_extra_dims=self.axis.ndim)
        other.transform += transform_list.inverse
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
        return np.prod(np.array(self.shape))

    @property
    def num_wavlength(self):
        return self.grid_shape[self.axis.wavelength]

    def _input_grid(self, axis: int) -> u.Quantity:
        grid = self.input_grids[axis]
        return np.broadcast_to(grid, self.shape + grid.shape[~0:], subok=True)

    @property
    def input_wavelength(self) -> u.Quantity:
        return self._input_grid(self.axis.wavelength)

    @property
    def input_field_x(self) -> u.Quantity:
        return self._input_grid(self.axis.field_x)

    @property
    def input_field_y(self) -> u.Quantity:
        return self._input_grid(self.axis.field_y)

    @property
    def input_pupil_x(self) -> u.Quantity:
        return self._input_grid(self.axis.pupil_x)

    @property
    def input_pupil_y(self) -> u.Quantity:
        return self._input_grid(self.axis.pupil_y)

    @property
    def mask(self) -> np.ndarray:
        return self.vignetted_mask & self.error_mask

    @property
    def field_mesh_input(self) -> vector.Vector2D:
        fx, fy = self.input_grids[self.axis.field_x], self.input_grids[self.axis.field_y]
        return vector.Vector2D(
            x=fx[..., :, np.newaxis],
            y=fy[..., np.newaxis, :],
        )

    @property
    def position_pupil_avg(self) -> vector.Vector3D:
        axes = (self.axis.pupil_x, self.axis.pupil_y)
        # mask = np.broadcast_to(self.mask[..., None], self.position.shape)
        # avg = np.ma.average(a=self.position, weights=self.mask, axis=axes, ) << self.position.unit
        # return np.expand_dims(avg, axis=axes)
        p = self.position.copy()
        p[~self.mask] = np.nan
        return np.nanmean(p, axis=axes, keepdims=True)

    @property
    def position_pupil_relative(self) -> vector.Vector3D:
        return self.position - self.position_pupil_avg

    def distortion(self, polynomial_degree: int = 1) -> Distortion:
        return Distortion(
            wavelength=self.input_wavelength[..., np.newaxis, np.newaxis, :],
            spatial_mesh_input=vector.Vector2D(
                x=self.input_field_x[..., :, np.newaxis, np.newaxis],
                y=self.input_field_y[..., np.newaxis, :, np.newaxis],
            ),
            spatial_mesh_output=self.position_pupil_avg[..., 0, 0, :].xy,
            mask=self.mask.any((self.axis.pupil_x, self.axis.pupil_y)),
            polynomial_degree=polynomial_degree
        )

    def vignetting(self, polynomial_degree: int = 1) -> Vignetting:
        counts = self.mask.sum((self.axis.pupil_x, self.axis.pupil_y))
        return Vignetting(
            wavelength=self.input_wavelength[..., np.newaxis, np.newaxis, :],
            spatial_mesh=vector.Vector2D(
                x=self.input_field_x[..., :, np.newaxis, np.newaxis],
                y=self.input_field_y[..., np.newaxis, :, np.newaxis],
            ),
            unvignetted_percent=100 * counts / counts.max() * u.percent,
            mask=self.mask.any((self.axis.pupil_x, self.axis.pupil_y)),
            polynomial_degree=polynomial_degree,
        )

    def aberration(
            self,
            distortion_polynomial_degree: int = 1,
            vignetting_polynomial_degree: int = 1,
    ) -> Aberration:
        return Aberration(
            distortion=self.distortion(polynomial_degree=distortion_polynomial_degree),
            vignetting=self.vignetting(polynomial_degree=vignetting_polynomial_degree)
        )

    def view(self) -> 'Rays':
        other = super().view()  # type: Rays
        other.wavelength = self.wavelength
        other.position = self.position
        other.direction = self.direction
        other.surface_normal = self.surface_normal
        other.index_of_refraction = self.index_of_refraction
        other.vignetted_mask = self.vignetted_mask
        other.error_mask = self.error_mask
        other.input_grids = self.input_grids
        return other

    def copy(self) -> 'Rays':
        other = super().copy()  # type: Rays
        other.wavelength = self.wavelength.copy()
        other.position = self.position.copy()
        other.direction = self.direction.copy()
        other.surface_normal = self.surface_normal.copy()
        other.index_of_refraction = self.index_of_refraction.copy()
        other.vignetted_mask = self.vignetted_mask.copy()
        other.error_mask = self.error_mask.copy()
        other.input_grids = [g.copy() for g in self.input_grids]
        return other

    @property
    def spot_size_rms(self):
        position = self.position_pupil_relative
        r = position.xy.length
        r2 = np.square(r)
        pupil_axes = self.axis.pupil_x, self.axis.pupil_y
        sz = np.sqrt(np.ma.average(r2.value, axis=pupil_axes, weights=self.mask) << r2.unit)
        mask = self.mask.any(pupil_axes)
        sz[~mask] = 0
        return sz

    def plot_spot_size_vs_field(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
    ) -> typ.MutableSequence[plt.Axes]:
        if axs is None:
            fig, axs = plt.subplots(ncols=self.num_wavlength)
        else:
            fig = axs[0].figure

        wavelength = self.input_wavelength
        field_x, field_y = self.input_field_x, self.input_field_y
        sizes = self.spot_size_rms
        if config_index is not None:
            field_x, field_y = field_x[config_index], field_y[config_index]
            wavelength = wavelength[config_index]
            sizes = sizes[config_index]

        vmin, vmax = sizes.min(), sizes.max()

        # for ax, wavl, sz in zip(axs, wavelength, sizes):
        for i in range(len(axs)):
            axs[i].set_title(fmt.quantity(wavelength[i]))
            sl = [slice(None)] * sizes.ndim
            sl[self.axis.wavelength] = i
            img = axs[i].imshow(
                X=sizes[sl].T.value,
                vmin=vmin.value,
                vmax=vmax.value,
                origin='lower',
                extent=[field_x[0].value, field_x[~0].value, field_y[0].value, field_y[~0].value],
            )
            axs[i].set_xlabel('input $x$ ' + '(' + "{0:latex}".format(field_x.unit) + ')')

        axs[0].set_ylabel('input $y$ ' + '(' + "{0:latex}".format(field_y.unit) + ')')

        fig.colorbar(img, ax=axs, label='RMS spot radius (' + '{0:latex}'.format(sizes.unit) + ')')

        return axs

    def pupil_hist2d(
            self,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limits: typ.Optional[typ.Tuple[typ.Tuple[int, int], typ.Tuple[int, int]]] = None,
            use_vignetted: bool = False,
            relative_to_centroid: typ.Tuple[bool, bool] = (False, False),
    ) -> typ.Tuple[np.ndarray, u.Quantity, u.Quantity]:

        if isinstance(bins, int):
            bins = (bins, bins)

        if not use_vignetted:
            mask = self.mask
        else:
            mask = self.error_mask

        position = self.position.copy()
        position_rel = self.position_pupil_relative
        if relative_to_centroid[vector.ix]:
            position.x = position_rel.x
        if relative_to_centroid[vector.iy]:
            position.y = position_rel.y

        if limits is None:
            px = position.x[mask]
            py = position.y[mask]
            limits = (
                (np.nanmin(px).value, np.nanmax(px).value),
                (np.nanmin(py).value, np.nanmax(py).value),
            )

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

        hist_flat = hist.reshape((-1, ) + hist.shape[~(self.axis.ndim - 1):])
        edges_x_flat = edges_x.reshape((-1, ) + edges_x.shape[~(self.axis.ndim - 1):])
        edges_y_flat = edges_y.reshape((-1, ) + edges_y.shape[~(self.axis.ndim - 1):])
        position_flat = position.reshape((-1,) + position.shape[~(self.axis.ndim - 1):])
        mask_flat = mask.reshape((-1,) + mask.shape[~(self.axis.ndim - 1):])

        for c in range(hist_flat.shape[0]):
            for i in range(hist_flat.shape[self.axis.field_x]):
                for j in range(hist_flat.shape[self.axis.field_y]):
                    for w in range(hist_flat.shape[self.axis.wavelength]):
                        cijw = [slice(None)] * hist_flat.ndim
                        cijw[0] = c
                        cijw[self.axis.field_x] = i
                        cijw[self.axis.field_y] = j
                        cijw[self.axis.wavelength] = w
                        cijwx = cijw.copy()
                        cijwx[self.axis.pupil_y] = 0
                        hist_flat[cijw], edges_x_flat[cijwx], edges_y_flat[cijw] = np.histogram2d(
                            x=position_flat[cijw].x.flatten().value,
                            y=position_flat[cijw].y.flatten().value,
                            bins=bins,
                            weights=mask_flat[cijw].flatten(),
                            range=limits,
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

        with astropy.visualization.quantity_support():
            scatter = ax.scatter(
                x=attr_x[mask],
                y=attr_y[mask],
                c=self.colormesh(color_axis)[mask].value,
            )
            try:
                ax.legend(
                    handles=scatter.legend_elements(num=self.input_grids[color_axis].flatten())[0],
                    labels=list(self.grid_labels(color_axis).flatten()),
                    loc='center left',
                    bbox_to_anchor=(1.0, 0.5),
                )
            except ValueError:
                pass

        return ax

    def plot_pupil_hist2d_vs_field(
            self,
            config_index: int = 0,
            wavlen_index: int = 0,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limits: typ.Optional[typ.Tuple[typ.Tuple[int, int], typ.Tuple[int, int]]] = None,
            use_vignetted: bool = False,
            relative_to_centroid: typ.Tuple[bool, bool] = (True, True),
            norm: typ.Optional[matplotlib.colors.Normalize] = None,
    ) -> plt.Figure:

        field_x = self.input_grids[self.axis.field_x]
        field_y = self.input_grids[self.axis.field_y]

        hist, edges_x, edges_y = self.pupil_hist2d(
            bins=bins,
            limits=limits,
            use_vignetted=use_vignetted,
            relative_to_centroid=relative_to_centroid,
        )

        fig, axs = plt.subplots(
            nrows=self.grid_shape[self.axis.field_x],
            ncols=self.grid_shape[self.axis.field_y],
            sharex='all',
            sharey='all',
            squeeze=False,
        )

        if hist.ndim > self.axis.ndim:
            hist, edges_x, edges_y = hist[config_index], edges_x[config_index], edges_y[config_index]

        for i, axs_i in enumerate(axs):
            for j, axs_ij in enumerate(axs_i):
                axs_ij.invert_xaxis()
                cwji = [slice(None)] * hist.ndim
                cwji[self.axis.wavelength] = wavlen_index
                cwji[self.axis.field_x] = j
                cwji[self.axis.field_y] = i
                # cwji = config_index, wavlen_index, j, i
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
                    aspect='auto',
                    origin='lower',
                    vmin=hist[w].min(),
                    vmax=hist[w].max(),
                    norm=norm,
                )
                if i == 0:
                    axs_ij.set_xlabel('{0.value:0.0f} {0.unit:latex}'.format(field_x[j]))
                    axs_ij.xaxis.set_label_position('top')
                elif i == len(axs) - 1:
                    axs_ij.set_xlabel(edges_x.unit)

                if j == 0:
                    axs_ij.set_ylabel(edges_y.unit)
                elif j == len(axs_i) - 1:
                    axs_ij.set_ylabel('{0.value:0.0f} {0.unit:latex}'.format(field_y[i]))
                    axs_ij.yaxis.set_label_position('right')

                axs_ij.tick_params(axis='both', labelsize=8)

        wavelength = self.input_grids[self.axis.wavelength]
        if wavelength.ndim > 1:
            wavelength = wavelength[config_index]
        wavl_str = wavelength[wavlen_index]
        wavl_str = '{0.value:0.3f} {0.unit:latex}'.format(wavl_str)
        fig.suptitle('configuration = ' + str(config_index) + ', wavelength = ' + wavl_str)
        fig.colorbar(img, ax=axs, fraction=0.05)

        return fig


class RaysList(
    collections.UserList,
    typ.List[Rays],
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
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[str, str] = ('x', 'y'),
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            color_axis: int = Rays.axis.wavelength,
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

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

        grid = img_rays.input_grids[color_axis]
        grid = np.broadcast_to(grid, img_rays.shape + grid.shape[~0:])
        grid = grid.flatten()
        grid_min, grid_max = grid.min(axis=~0, keepdims=True), grid.max(axis=~0, keepdims=True)
        grid_delta = grid_max - grid_min
        ngrid = (grid - grid_min) / grid_delta
        ngrid[~np.isfinite(ngrid)] = 0.5
        colors = plt.cm.viridis(ngrid)
        labels = img_rays.grid_labels(color_axis)
        labels = np.broadcast_to(labels, img_rays.shape + labels.shape[~0:])
        labels = labels.flatten()

        intercepts = np.moveaxis(intercepts, color_axis, img_rays.ndim + 1)
        mask = np.moveaxis(mask, color_axis, img_rays.ndim)

        new_shape = intercepts.shape[:1] + (-1,) + intercepts.shape[~(img_rays.axis.ndim - 2):]
        intercepts = intercepts.reshape(new_shape)
        mask = mask.reshape((-1,) + mask.shape[~(img_rays.axis.ndim - 2):])

        intercepts = np.moveaxis(intercepts, ~(img_rays.axis.ndim - 1), 0)
        mask = np.moveaxis(mask, ~(img_rays.axis.ndim - 1), 0)

        for intercept_c, mask_c, color, label in zip(intercepts, mask, colors, labels):
            ax.plot(
                intercept_c[:, mask_c].get_component(components[0]),
                intercept_c[:, mask_c].get_component(components[1]),
                color=color,
                label=label,
            )

        handles, labels = ax.get_legend_handles_labels()
        label_dict = dict(zip(labels, handles))
        ax.legend(label_dict.values(), label_dict.keys(), loc='center left', bbox_to_anchor=(1.0, 0.5))

        return ax

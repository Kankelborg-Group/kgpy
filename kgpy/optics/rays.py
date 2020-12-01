import dataclasses
import collections
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import astropy.units as u
import astropy.visualization
import astropy.modeling
# import kgpy.transform.rigid.transform_list
from kgpy import mixin, vector, transform, format as fmt
from kgpy.vector import x, y, z, ix, iy, iz
from .aberration import Distortion, Vignetting

__all__ = ['Rays', 'RaysList']


class CAxis(mixin.AutoAxis):
    def __init__(self):
        super().__init__()
        self.components = self.auto_axis_index()


class Axis(mixin.AutoAxis):
    def __init__(self):
        super().__init__()
        self.pupil_y = self.auto_axis_index()
        self.pupil_x = self.auto_axis_index()
        self.field_y = self.auto_axis_index()
        self.field_x = self.auto_axis_index()
        self.wavelength = self.auto_axis_index()

    @property
    def latex_names(self) -> typ.List[str]:
        return ['$\\lambda$', '$f_x$', '$f_y$', '$p_x$', '$p_y$']


class VAxis(Axis, CAxis):
    pass


@dataclasses.dataclass
class Rays(transform.rigid.Transformable):
    axis = Axis()
    vaxis = VAxis()

    wavelength: u.Quantity = dataclasses.field(default_factory=lambda: [[0]] * u.nm)
    position: u.Quantity = dataclasses.field(default_factory=lambda: [[0, 0, 0]] * u.mm)
    direction: u.Quantity = dataclasses.field(default_factory=lambda: [[0, 0, 1]] * u.dimensionless_unscaled)
    polarization: u.Quantity = dataclasses.field(default_factory=lambda: [[1, 0]] * u.dimensionless_unscaled)
    surface_normal: u.Quantity = dataclasses.field(default_factory=lambda: [[0, 0, -1]] * u.dimensionless_unscaled)
    index_of_refraction: u.Quantity = dataclasses.field(default_factory=lambda: [[1]] * u.dimensionless_unscaled)
    vignetted_mask: np.ndarray = np.array([True])
    error_mask: np.ndarray = np.array([True])
    input_grids: typ.List[typ.Optional[u.Quantity]] = dataclasses.field(
        default_factory=lambda: [None, None, None, None, None],
    )

    @property
    def field_angles(self) -> u.Quantity:
        angles = np.arcsin(self.direction)[vector.xy] << u.rad
        return angles[..., ::-1]

    @classmethod
    def from_field_angles(
            cls,
            wavelength_grid: u.Quantity,
            position: u.Quantity,
            field_grid_x: u.Quantity,
            field_grid_y: u.Quantity,
            pupil_grid_x: typ.Optional[u.Quantity] = None,
            pupil_grid_y: typ.Optional[u.Quantity] = None,
    ) -> 'Rays':

        wavelength = np.expand_dims(wavelength_grid, cls.vaxis.perp_axes(cls.vaxis.wavelength))
        field_x = np.expand_dims(field_grid_x, cls.vaxis.perp_axes(cls.vaxis.field_x))
        field_y = np.expand_dims(field_grid_y, cls.vaxis.perp_axes(cls.vaxis.field_y))
        wavelength, field_x, field_y = np.broadcast_arrays(wavelength, field_x, field_y, subok=True)

        position, _ = np.broadcast_arrays(position, wavelength, subok=True)

        direction = np.zeros(position.shape)
        direction[z] = 1
        direction = transform.rigid.TiltX(field_y[..., 0])(direction)
        direction = transform.rigid.TiltY(field_x[..., 0])(direction)

        return cls(
            wavelength=wavelength,
            position=position,
            direction=direction,
            input_grids=[wavelength_grid, field_grid_x, field_grid_y, pupil_grid_x, pupil_grid_y],
        )

    @classmethod
    def from_field_positions(
            cls,
            wavelength_grid: u.Quantity,
            direction: u.Quantity,
            field_grid_x: u.Quantity,
            field_grid_y: u.Quantity,
            pupil_grid_x: typ.Optional[u.Quantity] = None,
            pupil_grid_y: typ.Optional[u.Quantity] = None,
    ):
        wavelength = np.expand_dims(wavelength_grid, cls.vaxis.perp_axes(cls.vaxis.wavelength))
        pupil_x = np.expand_dims(pupil_grid_x, cls.vaxis.perp_axes(cls.vaxis.pupil_x))
        pupil_y = np.expand_dims(pupil_grid_y, cls.vaxis.perp_axes(cls.vaxis.pupil_y))
        wavelength, pupil_x, pupil_y = np.broadcast_arrays(wavelength, pupil_x, pupil_y, subok=True)

        direction, _ = np.broadcast_arrays(direction, wavelength, subok=True)

        position = vector.from_components(x=field_grid_x[..., None, None, None], y=field_grid_y[..., None, None])
        position, _ = np.broadcast_arrays(position, wavelength, subok=True)

        return cls(
            wavelength=wavelength,
            position=position,
            direction=direction,
            input_grids=[wavelength_grid, field_grid_x, field_grid_y, pupil_grid_x, pupil_grid_y],
        )

    def apply_transform_list(self, transform_list: transform.rigid.TransformList) -> 'Rays':
        other = self.copy()
        other.position = transform_list(other.position, num_extra_dims=5)
        other.direction = transform_list(other.direction, translate=False, num_extra_dims=5)
        other.surface_normal = transform_list(other.surface_normal, translate=False, num_extra_dims=5)
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
            self.wavelength[x],
            self.position[x],
            self.direction[x],
        ).shape

    @property
    def vector_grid_shape(self) -> typ.Tuple[int, ...]:
        return self.grid_shape + (3,)

    @property
    def scalar_grid_shape(self) -> typ.Tuple[int, ...]:
        return self.grid_shape + (1,)

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.vector_grid_shape[:~(self.vaxis.ndim - 1)]

    @property
    def ndim(self):
        return len(self.shape)

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
    def field_mesh_input(self) -> u.Quantity:
        fx, fy = self.input_grids[self.axis.field_x], self.input_grids[self.axis.field_y]
        return vector.from_components(x=fx[..., :, None], y=fy[..., None, :], use_z=False)

    @property
    def position_pupil_avg(self) -> u.Quantity:
        axes = (self.vaxis.pupil_x, self.vaxis.pupil_y)
        mask = np.broadcast_to(self.mask[..., None], self.position.shape)
        avg = np.ma.average(a=self.position.value, weights=mask, axis=axes, ) << self.position.unit
        return np.expand_dims(avg, axis=axes)

    @property
    def position_pupil_relative(self) -> u.Quantity:
        return self.position - self.position_pupil_avg

    def distortion(self, polynomial_degree: int = 1) -> Distortion:
        return Distortion(
            wavelength=self.input_wavelength[..., :, None, None],
            spatial_mesh_input=vector.from_components(
                x=self.input_field_x[..., None, :, None],
                y=self.input_field_y[..., None, None, :],
                use_z=False,
            ),
            spatial_mesh_output=self.position_pupil_avg[..., 0, 0, :][vector.xy],
            mask=self.mask.any((self.axis.pupil_x, self.axis.pupil_y)),
            polynomial_degree=polynomial_degree
        )

    def vignetting(self, polynomial_degree: int = 1):
        counts = self.mask.sum((self.axis.pupil_x, self.axis.pupil_y))
        return Vignetting(
            wavelength=self.input_wavelength[..., :, None, None],
            spatial_mesh=vector.from_components(
                x=self.input_field_x[..., None, :, None],
                y=self.input_field_y[..., None, None, :],
                use_z=False,
            ),
            unvignetted_percent=100 * counts / counts.max() * u.percent,
            mask=self.mask.any((self.axis.pupil_x, self.axis.pupil_y)),
            polynomial_degree=polynomial_degree,
        )

    def copy(self) -> 'Rays':
        other = super().copy()  # type: Rays
        other.wavelength = self.wavelength.copy()
        other.position = self.position.copy()
        other.direction = self.direction.copy()
        other.polarization = self.polarization.copy()
        other.surface_normal = self.surface_normal.copy()
        other.index_of_refraction = self.index_of_refraction.copy()
        other.vignetted_mask = self.vignetted_mask.copy()
        other.error_mask = self.error_mask.copy()
        other.input_grids = self.input_grids
        # other.input_grids = copy.deepcopy(self.input_grids)
        return other

    @property
    def spot_size_rms(self):
        position = self.position_pupil_relative
        r = vector.length(position[vector.xy], keepdims=False)
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

        for ax, wavl, sz in zip(axs, wavelength, sizes):
            ax.set_title(fmt.quantity(wavl))
            img = ax.imshow(
                X=sz.T.value,
                vmin=vmin.value,
                vmax=vmax.value,
                origin='lower',
                extent=[field_x[0].value, field_x[~0].value, field_y[0].value, field_y[~0].value],
            )
            ax.set_xlabel('input $x$ ' + '(' + "{0:latex}".format(field_x.unit) + ')')

        axs[0].set_ylabel('input $y$ ' + '(' + "{0:latex}".format(field_y.unit) + ')')

        fig.colorbar(img, ax=axs, label='RMS spot radius (' + '{0:latex}'.format(sizes.unit) + ')')

        return axs

    def pupil_hist2d(
            self,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limits: typ.Optional[typ.Tuple[typ.Tuple[int, int], typ.Tuple[int, int]]] = None,
            use_vignetted: bool = False,
            relative_to_centroid: typ.Tuple[bool, bool] = (False, False),
    ) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if isinstance(bins, int):
            bins = (bins, bins)

        if not use_vignetted:
            mask = self.mask
        else:
            mask = self.error_mask

        position = self.position.copy()
        position_rel = self.position_pupil_relative
        if relative_to_centroid[vector.ix]:
            position[x] = position_rel[x]
        if relative_to_centroid[vector.iy]:
            position[y] = position_rel[y]

        if limits is None:
            px = position[x][mask]
            py = position[y][mask]
            limits = (
                (px.min().value, px.max().value),
                (py.min().value, py.max().value),
            )

        base_shape = self.shape + self.grid_shape[self.axis.wavelength:self.axis.field_y + 1]
        hist = np.empty(base_shape + tuple(bins))
        edges_x = np.empty(base_shape + (bins[vector.ix] + 1,))
        edges_y = np.empty(base_shape + (bins[vector.iy] + 1,))

        if not self.shape:
            position = position[None, ...]
            mask = mask[None, ...]
            hist, edges_x, edges_y = hist[None, ...], edges_x[None, ...], edges_y[None, ...]

        for c, p_c in enumerate(position):
            for w, p_cw in enumerate(p_c):
                for i, p_cwi in enumerate(p_cw):
                    for j, p_cwij in enumerate(p_cwi):
                        cwij = c, w, i, j
                        hist[cwij], edges_x[cwij], edges_y[cwij] = np.histogram2d(
                            x=p_cwij[x].flatten().value,
                            y=p_cwij[y].flatten().value,
                            bins=bins,
                            weights=mask[cwij].flatten(),
                            range=limits,
                        )

        unit = self.position.unit
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
        if ax is None:
            _, ax = plt.subplots()

        if plot_vignetted:
            mask = self.error_mask
        else:
            mask = self.mask
        mask = np.broadcast_to(mask, self.position[x].shape)

        with astropy.visualization.quantity_support():
            scatter = ax.scatter(
                x=self.position[x][mask],
                y=self.position[y][mask],
                c=self.colormesh(color_axis)[mask].value,
            )
            try:
                ax.legend(
                    handles=scatter.legend_elements(num=self.input_grids[color_axis].flatten())[0],
                    labels=list(self.grid_labels(color_axis).flatten()),
                    loc='upper right',
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

        for i, axs_i in enumerate(axs):
            for j, axs_ij in enumerate(axs_i):
                axs_ij.invert_xaxis()
                cwji = config_index, wavlen_index, j, i
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
                    vmin=hist[config_index, wavlen_index].min(),
                    vmax=hist[config_index, wavlen_index].max(),
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
        wavelength = self.input_grids[self.axis.wavelength]
        if wavelength.ndim == 1:
            wavelength = wavelength[None, ...]
        wavl_str = wavelength[config_index, wavlen_index]
        wavl_str = '{0.value:0.3f} {0.unit:latex}'.format(wavl_str)
        fig.suptitle('configuration = ' + str(config_index) + ', wavelength = ' + wavl_str)
        fig.colorbar(img, ax=axs, fraction=0.05)

        return fig


class RaysList(
    collections.UserList,
    typ.List[Rays],
):
    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
            color_axis: int = Rays.axis.wavelength,
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if rigid_transform is None:
            rigid_transform = transform.rigid.TransformList()

        intercepts = []
        for rays in self:
            rays_transform = rigid_transform + rays.transform
            intercepts.append(rays_transform(rays.position, num_extra_dims=5))
        intercepts = u.Quantity(intercepts)

        img_rays = self[~0]

        color_axis = (color_axis % img_rays.axis.ndim) - img_rays.axis.ndim

        if plot_vignetted:
            mask = img_rays.error_mask
        else:
            mask = img_rays.mask
        mask = np.broadcast_to(mask, img_rays.grid_shape)

        grid = img_rays.input_grids[color_axis].flatten()
        colors = plt.cm.viridis((grid - grid.min()) / (grid.max() - grid.min()))
        labels = img_rays.grid_labels(color_axis).flatten()

        intercepts = np.moveaxis(intercepts, color_axis - 1, img_rays.ndim + 1)
        mask = np.moveaxis(mask, color_axis, img_rays.ndim)

        new_shape = intercepts.shape[0:1] + (-1,) + grid.shape + intercepts.shape[~(img_rays.vaxis.ndim - 2):]
        intercepts = intercepts.reshape(new_shape)
        mask = mask.reshape((-1,) + grid.shape + mask.shape[~(img_rays.axis.ndim - 2):])

        intercepts = np.moveaxis(intercepts, ~(img_rays.vaxis.ndim - 1), 0)
        mask = np.moveaxis(mask, ~(img_rays.axis.ndim - 1), 0)

        for intercept_c, mask_c, color, label in zip(intercepts, mask, colors, labels):
            ax.plot(
                intercept_c[:, mask_c, components[0]],
                intercept_c[:, mask_c, components[1]],
                color=color,
                label=label,
            )

        ax.set_xlim(right=1.1 * ax.get_xlim()[1])
        handles, labels = ax.get_legend_handles_labels()
        label_dict = dict(zip(labels, handles))
        ax.legend(label_dict.values(), label_dict.keys())

        return ax

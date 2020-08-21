import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import astropy.units as u
import astropy.visualization
import kgpy.vector
from kgpy.vector import x, y, z, ix, iy, iz
from . import coordinate

__all__ = ['Rays']


class AutoAxis:

    def __init__(self):
        super().__init__()
        self.ndim = 0
        self.all = []

    def auto_axis_index(self):
        i = ~self.ndim
        self.all.append(i)
        self.ndim += 1
        return i

    def perp_axes(self, axis: int) -> typ.Tuple[int, ...]:
        axes = self.all.copy()
        axes = [a % self.ndim for a in axes]
        axes.remove(axis % self.ndim)
        return tuple([a - self.ndim for a in axes])


class CAxis(AutoAxis):
    def __init__(self):
        super().__init__()
        self.components = self.auto_axis_index()


class Axis(AutoAxis):
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
class Rays:

    axis = Axis()
    vaxis = VAxis()

    wavelength: u.Quantity
    position: u.Quantity
    direction: u.Quantity
    polarization: u.Quantity = None
    surface_normal: u.Quantity = None
    index_of_refraction: u.Quantity = None
    field_mask: np.ndarray = None
    vignetted_mask: np.ndarray = None
    error_mask: np.ndarray = None
    input_grids: typ.List[typ.Optional[u.Quantity]] = dataclasses.field(
        default_factory=lambda: [None, None, None, None, None],
    )

    def __post_init__(self):
        if self.polarization is None:
            self.polarization = np.zeros(self.vector_grid_shape) << u.dimensionless_unscaled
            self.polarization[z] = 1
        if self.surface_normal is None:
            self.surface_normal = np.zeros(self.vector_grid_shape) << u.dimensionless_unscaled
            self.surface_normal[z] = 1
        if self.index_of_refraction is None:
            self.index_of_refraction = np.ones(self.scalar_grid_shape) << u.dimensionless_unscaled
        if self.vignetted_mask is None:
            self.vignetted_mask = np.ones(self.grid_shape, dtype=np.bool)
        if self.error_mask is None:
            self.error_mask = np.ones(self.grid_shape, dtype=np.bool)

    @classmethod
    def from_field_angles(
            cls,
            wavelength_grid: u.Quantity,
            position: u.Quantity,
            field_grid_x: u.Quantity,
            field_grid_y: u.Quantity,
            field_mask_func: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], np.ndarray]] = None,
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
        direction = kgpy.vector.rotate_x(direction, field_y[..., 0])
        direction = kgpy.vector.rotate_y(direction, field_x[..., 0])

        mask = field_mask_func(np.arcsin(direction[x]) << u.rad, np.arcsin(direction[y]) << u.rad)

        return cls(
            wavelength=wavelength,
            position=position,
            direction=direction,
            field_mask=mask,
            input_grids=[wavelength_grid, field_grid_x, field_grid_y, pupil_grid_x, pupil_grid_y],
        )

    @classmethod
    def from_field_positions(
            cls,
            wavelength_grid: u.Quantity,
            direction: u.Quantity,
            field_grid_x: u.Quantity,
            field_grid_y: u.Quantity,
            field_mask_func: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], np.ndarray]] = None,
            pupil_grid_x: typ.Optional[u.Quantity] = None,
            pupil_grid_y: typ.Optional[u.Quantity] = None,
    ):
        wavelength = np.expand_dims(wavelength_grid, cls.vaxis.perp_axes(cls.vaxis.wavelength))
        pupil_x = np.expand_dims(pupil_grid_x, cls.vaxis.perp_axes(cls.vaxis.pupil_x))
        pupil_y = np.expand_dims(pupil_grid_y, cls.vaxis.perp_axes(cls.vaxis.pupil_y))
        wavelength, pupil_x, pupil_y = np.broadcast_arrays(wavelength, pupil_x, pupil_y, subok=True)

        direction, _ = np.broadcast_arrays(direction, wavelength, subok=True)

        position = kgpy.vector.from_components(ax=field_grid_x[..., None, None, None], ay=field_grid_y[..., None, None])
        position, _ = np.broadcast_arrays(position, wavelength, subok=True)
        mask = field_mask_func(position[x], position[y])

        return cls(
            wavelength=wavelength,
            position=position,
            direction=direction,
            field_mask=mask,
            input_grids=[wavelength_grid, field_grid_x, field_grid_y, pupil_grid_x, pupil_grid_y],
        )

    def plane_intersection(self, plane_position: u.Quantity, plane_normal: u.Quantity):
        pass

    def apply_transform(self, transform: coordinate.Transform) -> 'Rays':
        other = self.copy()
        other.position = transform(other.position, num_extra_dims=5)
        other.direction = transform(other.direction, use_translations=False, num_extra_dims=5)
        other.surface_normal = transform(other.surface_normal, use_translations=False, num_extra_dims=5)
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
        return self.grid_shape + (3, )

    @property
    def scalar_grid_shape(self) -> typ.Tuple[int, ...]:
        return self.grid_shape + (1, )

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.vector_grid_shape[:~(self.vaxis.ndim - 1)]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def mask(self) -> np.ndarray:
        return self.vignetted_mask & self.error_mask & self.field_mask

    def copy(self) -> 'Rays':
        return Rays(
            wavelength=self.wavelength.copy(),
            position=self.position.copy(),
            direction=self.direction.copy(),
            surface_normal=self.surface_normal.copy(),
            polarization=self.polarization.copy(),
            index_of_refraction=self.index_of_refraction.copy(),
            input_grids=self.input_grids.copy(),
            field_mask=self.field_mask.copy(),
            vignetted_mask=self.vignetted_mask.copy(),
            error_mask=self.error_mask.copy(),
        )

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
            mask = self.error_mask & self.field_mask

        position = self.position.copy()
        if relative_to_centroid[kgpy.vector.ix]:
            position[x] -= np.mean(position[x].value, axis=self.axis.pupil_x, keepdims=True) << position.unit
        if relative_to_centroid[kgpy.vector.iy]:
            position[y] -= np.mean(position[y].value, axis=self.axis.pupil_y, keepdims=True) << position.unit

        if limits is None:
            px = position[x][mask]
            py = position[y][mask]
            limits = (
                (px.min().value, px.max().value),
                (py.min().value, py.max().value),
            )

        base_shape = self.shape + self.grid_shape[self.axis.wavelength:self.axis.field_y + 1]
        hist = np.empty(base_shape + tuple(bins))
        edges_x = np.empty(base_shape + (bins[kgpy.vector.ix] + 1,))
        edges_y = np.empty(base_shape + (bins[kgpy.vector.iy] + 1,))

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
            mask = self.error_mask & self.field_mask
        else:
            mask = self.mask

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
            relative_to_centroid: typ.Tuple[bool, bool] = (False, False),
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
                    axs_ij.set_xlabel('{0.value:0.2f} {0.unit:latex}'.format(field_x[j]))
                    axs_ij.xaxis.set_label_position('top')
                elif i == len(axs) - 1:
                    axs_ij.set_xlabel(edges_x.unit)

                if j == 0:
                    axs_ij.set_ylabel(edges_y.unit)
                elif j == len(axs_i) - 1:
                    axs_ij.set_ylabel('{0.value:0.2f} {0.unit:latex}'.format(field_y[i]))
                    axs_ij.yaxis.set_label_position('right')
        wavelength = self.input_grids[self.axis.wavelength]
        if wavelength.ndim == 1:
            wavelength = wavelength[None, ...]
        wavl_str = wavelength[config_index, wavlen_index]
        wavl_str = '{0.value:0.3f} {0.unit:latex}'.format(wavl_str)
        fig.suptitle('configuration = ' + str(config_index) + ', wavelength = ' + wavl_str)
        fig.colorbar(img, ax=axs, fraction=0.05)

        return fig



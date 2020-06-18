import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import kgpy.vector
from . import coordinate

__all__ = ['Rays']


class AutoAxis:

    def __init__(self):
        super().__init__()
        self.num_axes = 0

    def auto_axis_index(self):
        i = ~self.num_axes
        self.num_axes += 1
        return i


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


class VAxis(Axis, CAxis):
    pass


@dataclasses.dataclass
class Rays:

    axis = Axis()
    vaxis = VAxis()

    wavelength: u.Quantity
    position: u.Quantity
    direction: u.Quantity
    polarization: u.Quantity
    surface_normal: u.Quantity
    index_of_refraction: u.Quantity
    unvignetted_mask: np.ndarray
    error_mask: np.ndarray

    @classmethod
    def zeros(cls, shape: typ.Tuple[int, ...] = ()):
        vsh = shape + (3, )
        ssh = shape + (1, )

        direction = np.zeros(vsh) << u.dimensionless_unscaled
        polarization = np.zeros(vsh) << u.dimensionless_unscaled
        normal = np.zeros(vsh) << u.dimensionless_unscaled

        direction[kgpy.vector.z] = 1
        polarization[kgpy.vector.x] = 1
        normal[kgpy.vector.z] = 1

        return cls(
            wavelength=np.zeros(ssh) << u.nm,
            position=np.zeros(vsh) << u.mm,
            direction=direction,
            polarization=polarization,
            surface_normal=normal,
            index_of_refraction=np.ones(ssh) << u.dimensionless_unscaled,
            unvignetted_mask=np.zeros(shape, dtype=np.bool),
            error_mask=np.zeros(shape, dtype=np.bool),
        )

    def tilt_decenter(self, transform: coordinate.TiltDecenter) -> 'Rays':
        return type(self)(
            wavelength=self.wavelength.copy(),
            position=transform(self.position, num_extra_dims=5),
            direction=transform(self.direction, decenter=False, num_extra_dims=5),
            polarization=self.polarization.copy(),
            surface_normal=transform(self.surface_normal, decenter=False, num_extra_dims=5),
            index_of_refraction=self.index_of_refraction.copy(),
            unvignetted_mask=self.unvignetted_mask.copy(),
            error_mask=self.error_mask.copy(),
        )

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return np.broadcast(
            self.wavelength[..., 0],
            self.position[..., 0],
            self.direction[..., 0],
            self.surface_normal[..., 0],
            self.unvignetted_mask,
            self.error_mask,
            self.polarization[..., 0],
            self.index_of_refraction[..., 0],
        ).shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def px(self) -> u.Quantity:
        return self.position[kgpy.vector.x]

    @px.setter
    def px(self, value: u.Quantity):
        self.position[kgpy.vector.x] = value

    @property
    def py(self) -> u.Quantity:
        return self.position[kgpy.vector.y]

    @py.setter
    def py(self, value: u.Quantity):
        self.position[kgpy.vector.y] = value

    @property
    def pz(self) -> u.Quantity:
        return self.position[kgpy.vector.z]

    @pz.setter
    def pz(self, value: u.Quantity):
        self.position[kgpy.vector.z] = value

    @property
    def goodmask(self) -> np.ndarray:
        return self.unvignetted_mask & ~self.error_mask

    @property
    def gx(self) -> typ.Tuple[np.ndarray, int]:
        return self.goodmask, kgpy.vector.ix

    @property
    def gy(self):
        return self.goodmask, kgpy.vector.iy

    def copy(self) -> 'Rays':
        return Rays(
            wavelength=self.wavelength.copy(),
            position=self.position.copy(),
            direction=self.direction.copy(),
            surface_normal=self.surface_normal.copy(),
            unvignetted_mask=self.unvignetted_mask.copy(),
            error_mask=self.error_mask.copy(),
            polarization=self.polarization.copy(),
            index_of_refraction=self.index_of_refraction.copy(),
        )

    def pupil_hist2d(
            self,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limits: typ.Optional[typ.Tuple[typ.Tuple[int, int], typ.Tuple[int, int]]] = None,
            use_vignetted: bool = False,
            relative_to_centroid: bool = False,
    ) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if isinstance(bins, int):
            bins = (bins, bins)

        if not use_vignetted:
            mask = self.goodmask
        else:
            mask = np.broadcast_to(True, self.shape)

        position = self.position.copy()
        if relative_to_centroid:
            axes = (self.vaxis.pupil_x, self.vaxis.pupil_y)
            position -= np.mean(position.value, axis=axes, keepdims=True) << position.unit

        if limits is None:
            px = position[kgpy.vector.x]
            py = position[kgpy.vector.y]
            if not use_vignetted:
                px = px[self.goodmask]
                py = py[self.goodmask]
            limits = (
                (px.min().value, px.max().value),
                (py.min().value, py.max().value),
            )

        base_shape = self.shape[:~1]
        hist = np.empty(base_shape + tuple(bins))
        edges_x = np.empty(base_shape + (bins[kgpy.vector.ix] + 1,))
        edges_y = np.empty(base_shape + (bins[kgpy.vector.iy] + 1,))

        for c, p_c in enumerate(position):
            for w, p_cw in enumerate(p_c):
                for i, p_cwi in enumerate(p_cw):
                    for j, p_cwij in enumerate(p_cwi):
                        cwij = c, w, i, j
                        hist[cwij], edges_x[cwij], edges_y[cwij] = np.histogram2d(
                            x=p_cwij[kgpy.vector.x].flatten().value,
                            y=p_cwij[kgpy.vector.y].flatten().value,
                            bins=bins,
                            weights=mask[cwij].flatten(),
                            range=limits,
                        )

        unit = self.position.unit
        return hist, edges_x << unit, edges_y << unit

    def plot_pupil_hist2d_vs_field(
            self,
            config_index: int = 0,
            wavlen_index: int = 0,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limits: typ.Optional[typ.Tuple[typ.Tuple[int, int], typ.Tuple[int, int]]] = None,
            use_vignetted: bool = False,
            field_x: typ.Optional[u.Quantity] = None,
            field_y: typ.Optional[u.Quantity] = None,
    ) -> plt.Figure:

        if field_x is None:
            fax_x = (self.axis.field_y, self.axis.pupil_x, self.axis.pupil_y)
            field_x = self.direction[config_index, wavlen_index, ..., kgpy.vector.ix].mean(fax_x)
            field_x = np.arcsin(field_x) << u.rad
        if field_y is None:
            fax_y = (self.axis.field_x, self.axis.pupil_x, self.axis.pupil_y)
            field_y = self.direction[config_index, wavlen_index, ..., kgpy.vector.iy].mean(fax_y)
            field_y = np.arcsin(field_y) << u.rad

        hist, edges_x, edges_y = self.pupil_hist2d(
            bins=bins,
            limits=limits,
            use_vignetted=use_vignetted,
            relative_to_centroid=True,
        )

        fig, axs = plt.subplots(
            nrows=self.shape[self.axis.field_x],
            ncols=self.shape[self.axis.field_y],
            sharex='all',
            sharey='all',
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
                )
                if i == 0:
                    axs_ij.set_xlabel('{0.value:0.2f} {0.unit:latex}'.format(field_x[j].to(u.deg)))
                    axs_ij.xaxis.set_label_position('top')
                elif i == len(axs) - 1:
                    axs_ij.set_xlabel(edges_x.unit)

                if j == 0:
                    axs_ij.set_ylabel(edges_y.unit)
                elif j == len(axs_i) - 1:
                    axs_ij.set_ylabel('{0.value:0.2f} {0.unit:latex}'.format(field_y[i].to(u.deg)))
                    axs_ij.yaxis.set_label_position('right')

        wavl_str = np.unique(self.wavelength[config_index, wavlen_index]).squeeze()
        wavl_str = '{0.value:0.3f} {0.unit:latex}'.format(wavl_str)
        fig.suptitle('configuration = ' + str(config_index) + ', wavelength = ' + wavl_str)
        fig.colorbar(img, ax=axs, fraction=0.05)

        return fig



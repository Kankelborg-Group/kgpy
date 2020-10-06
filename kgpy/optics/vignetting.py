import typing as typ
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import polynomial, format as fmt
from kgpy.vector import x, y, z

__all__ = ['Vignetting']


@dataclasses.dataclass
class Vignetting:
    wavelength: u.Quantity
    spatial_mesh: u.Quantity
    unvignetted_percent: u.Quantity
    mask: np.ndarray
    polynomial_degree: int = 1

    def __call__(
            self,
            cube: np.ndarray,
            wavelength: u.Quantity,
            spatial_domain: u.Quantity,
            inverse: bool = False,
    ) -> np.ndarray:
        return Vignetting.apply_model(
            model=self.model(inverse=inverse),
            cube=cube,
            wavelength=wavelength,
            spatial_domain=spatial_domain,
        )

    @staticmethod
    def apply_model(
            model: polynomial.Polynomial3D,
            cube: np.ndarray,
            wavelength: u.Quantity,
            spatial_domain: u.Quantity,
    ):
        output_min, output_max = spatial_domain

        grid_x = np.linspace(output_min[x], output_max[x], cube.shape[~1])
        grid_y = np.linspace(output_min[y], output_max[y], cube.shape[~0])
        wavelength, grid_x, grid_y = np.broadcast_arrays(wavelength[..., None, None], grid_x[..., None], grid_y, subok=True)

        vig = model(wavelength, grid_x, grid_y).to(u.dimensionless_unscaled)[..., 0]

        return vig * cube

    def model(self, inverse: bool = False) -> polynomial.Polynomial3D:
        data = self.unvignetted_percent
        names_input, names_output = ['\\lambda', 'x', 'y'], ['$V$']
        if inverse:
            data = 1 / data
            names_output = ['$V\'$']

        model = polynomial.Polynomial3D.from_lstsq_fit(
            x=self.wavelength,
            y=self.spatial_mesh[x],
            z=self.spatial_mesh[y],
            data=data[..., None],
            mask=self.mask,
            degree=self.polynomial_degree,
        )
        model.component_names_input = names_input
        model.component_names_output = names_output
        return model

    def residual(
            self,
            other: typ.Optional['Vignetting'] = None,
            inverse: bool = False
    ) -> u.Quantity:
        if other is None:
            other = self
        data = other.unvignetted_percent
        if inverse:
            data = 1 / data
        model = self.model(inverse=inverse)
        residual = data[..., None] - model(x=other.wavelength, y=other.spatial_mesh[x], z=other.spatial_mesh[y])
        residual = residual[..., 0]
        residual[~other.mask] = 0
        return residual

    def plot_residual(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
            other: typ.Optional['Vignetting'] = None,
            inverse: bool = False,
            use_titles: bool = True,
            use_xlabels: bool = True,
    ) -> typ.MutableSequence[plt.Axes]:
        if other is None:
            other = self
        return Vignetting.plot(
            wavelength=other.wavelength,
            spatial_mesh=other.spatial_mesh,
            data=self.residual(other=other, inverse=inverse),
            axs=axs,
            config_index=config_index,
            data_name='residual',
            use_titles=use_titles,
            use_xlabels=use_xlabels,
        )

    def plot_unvignetted(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
            other: typ.Optional['Vignetting'] = None,
            inverse: bool = False,
            use_titles: bool = True,
            use_xlabels: bool = True,
    ) -> typ.MutableSequence[plt.Axes]:
        if other is None:
            other = self
        data = self.unvignetted_percent
        if inverse:
            data = 1 / data
        return Vignetting.plot(
            wavelength=other.wavelength,
            spatial_mesh=other.spatial_mesh,
            data=data,
            axs=axs,
            config_index=config_index,
            data_name='relative illumination',
            use_titles=use_titles,
            use_xlabels=use_xlabels,
        )

    @staticmethod
    def plot(
            wavelength: u.Quantity,
            spatial_mesh: u.Quantity,
            data: u.Quantity,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
            data_name: str = '',
            use_titles: bool = True,
            use_xlabels: bool = True,
    ) -> typ.MutableSequence[plt.Axes]:

        if config_index is not None:
            wavelength = wavelength[config_index]
            spatial_mesh = spatial_mesh[config_index]
            data = data[config_index]

        grid_shape = np.broadcast(wavelength, spatial_mesh[x], data).shape
        vgrid_shape = grid_shape + spatial_mesh.shape[~0:]
        wavelength = np.broadcast_to(wavelength, grid_shape, subok=True)
        spatial_mesh = np.broadcast_to(spatial_mesh, vgrid_shape, subok=True)
        data = np.broadcast_to(data, grid_shape, subok=True)

        if axs is None:
            fig, axs = plt.subplots(ncols=len(wavelength))
        else:
            fig = axs[0].figure

        vmin, vmax = data.min(), data.max()

        for ax, wavl, mesh, d in zip(axs, wavelength, spatial_mesh, data):
            if use_titles:
                ax.set_title(fmt.quantity(wavl[0, 0]))

            min_x, min_y = mesh[x].min(), mesh[y].min()
            max_x, max_y = mesh[x].max(), mesh[y].max()

            img = ax.imshow(
                X=d.value.T,
                vmin=vmin.value,
                vmax=vmax.value,
                origin='lower',
                extent=[min_x.value, max_x.value, min_y.value, max_y.value],
            )
            if use_xlabels:
                ax.set_xlabel('input $x$ ' + '(' + "{0:latex}".format(spatial_mesh.unit) + ')')

        axs[0].set_ylabel('input $y$ ' + '(' + "{0:latex}".format(spatial_mesh.unit) + ')')

        fig.colorbar(img, ax=axs, label=data_name + ' (' + '{0:latex}'.format(data.unit) + ')')

        return axs

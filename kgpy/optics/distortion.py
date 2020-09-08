import typing as typ
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import pandas
from kgpy import mixin, vector, format as fmt, polynomial
from kgpy.vector import x, y, z

__all__ = ['Distortion']


@dataclasses.dataclass
class Distortion:
    wavelength: u.Quantity
    spatial_mesh_input: u.Quantity
    spatial_mesh_output: u.Quantity
    mask: np.ndarray
    polynomial_degree: int = 1

    def model(self, inverse: bool = False) -> polynomial.Polynomial3D:
        if not inverse:
            mesh_input, mesh_output = self.spatial_mesh_input, self.spatial_mesh_output
            names_input, names_output = ['\\lambda', 'x', 'y'], ['$x\'$', '$y\'$']
        else:
            mesh_input, mesh_output = self.spatial_mesh_output, self.spatial_mesh_input
            names_input, names_output = ['\\lambda', 'x\'', 'y\''], ['$x$', '$y$']
        model = polynomial.Polynomial3D.from_lstsq_fit(
            x=self.wavelength,
            y=mesh_input[x],
            z=mesh_input[y],
            data=mesh_output,
            mask=self.mask,
            degree=self.polynomial_degree
        )
        model.component_names_input = names_input
        model.component_names_output = names_output
        return model

    def residual(self, inverse: bool = False):
        mesh_input, mesh_output = self.spatial_mesh_input, self.spatial_mesh_output
        if inverse:
            mesh_input, mesh_output = mesh_output, mesh_input
        model = self.model(inverse=inverse)
        return mesh_output - model(x=self.wavelength, y=mesh_input[x], z=mesh_input[y])

    def plot_residual(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
            inverse: bool = False,
            use_titles: bool = True,
            use_xlabels: bool = True,
    ) -> typ.MutableSequence[plt.Axes]:
        wavelength = self.wavelength
        mesh_input = self.spatial_mesh_input
        mesh_output = self.spatial_mesh_output
        if inverse:
            mesh_input, mesh_output = mesh_output, mesh_input
        mask = self.mask
        if config_index is not None:
            wavelength = wavelength[config_index]
            mesh_input = mesh_input[config_index]
            mesh_output = mesh_output[config_index]
            mask = mask[config_index]

        residual = vector.length(self.residual(inverse=inverse), keepdims=False)
        residual[~mask] = 0

        grid_shape = np.broadcast(wavelength, mesh_input[x], residual).shape
        vgrid_shape = grid_shape + mesh_input.shape[~0:]
        wavelength = np.broadcast_to(wavelength, grid_shape, subok=True)
        mesh_input = np.broadcast_to(mesh_input, vgrid_shape, subok=True)
        residual = np.broadcast_to(residual, grid_shape, subok=True)

        if axs is None:
            fig, axs = plt.subplots(ncols=len(wavelength))
        else:
            fig = axs[0].figure

        vmin, vmax = residual.min(), residual.max()

        for ax, wavl, mesh_in, res in zip(axs, wavelength, mesh_input, residual):
            if use_titles:
                ax.set_title(fmt.quantity(wavl[0, 0]))

            img = ax.imshow(
                X=res.value.T,
                vmin=vmin.value,
                vmax=vmax.value,
                origin='lower',
                extent=[
                    mesh_in[x].value.min(),
                    mesh_in[x].value.max(),
                    mesh_in[y].value.min(),
                    mesh_in[y].value.max(),
                ]
            )
            if use_xlabels:
                ax.set_xlabel('input $x$ ' + '(' + "{0:latex}".format(mesh_input.unit) + ')')

        axs[0].set_ylabel('input $y$ ' + '(' + "{0:latex}".format(mesh_input.unit) + ')')

        fig.colorbar(img, ax=axs, label='residual mag. (' + '{0:latex}'.format(residual.unit) + ')')

        return axs

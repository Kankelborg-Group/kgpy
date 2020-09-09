import typing as typ
import dataclasses
import numpy as np
import scipy.ndimage
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

    def residual(
            self,
            other: typ.Optional['Distortion'] = None,
            inverse: bool = False,
    ) -> u.Quantity:
        if other is None:
            other = self
        mesh_input, mesh_output = other.spatial_mesh_input, other.spatial_mesh_output
        if inverse:
            mesh_input, mesh_output = mesh_output, mesh_input
        model = self.model(inverse=inverse)
        residual = mesh_output - model(x=other.wavelength, y=mesh_input[x], z=mesh_input[y])
        residual[~other.mask] = 0
        return residual

    def distort_cube(
            self,
            cube: np.ndarray,
            wavelength: u.Quantity,
            spatial_domain_output: u.Quantity,
            spatial_samples_output: typ.Union[int, typ.Tuple[int, int]],
            inverse: bool = False,
            interp_order: int = 1,
            interp_prefilter: bool = False,
    ) -> np.ndarray:

        if isinstance(spatial_samples_output, int):
            spatial_samples_output = 2 * (spatial_samples_output,)
        spatial_samples_output = np.array(spatial_samples_output)

        output_min, output_max = spatial_domain_output

        output_grid_x = np.linspace(output_min[x], output_max[x], spatial_samples_output[x])[:, None]
        output_grid_y = np.linspace(output_min[y], output_max[y], spatial_samples_output[y])
        output_grid_x, output_grid_y = np.broadcast_arrays(output_grid_x, output_grid_y, subok=True)

        model = self.model(inverse=inverse)

        cube_flat = cube.reshape((-1,) + cube.shape[~2:])
        wavelength_flat = wavelength.reshape((-1,) + wavelength.shape[~0:])

        new_cube_flat = np.empty(cube_flat.shape[:~1] + tuple(spatial_samples_output))

        for i in range(cube_flat.shape[0]):
            for j in range(cube_flat.shape[1]):
                new_cube_flat[i, j] = scipy.ndimage.map_coordinates(
                    input=cube_flat[i, j],
                    coordinates=model(wavelength_flat[i, j], output_grid_x, output_grid_y),
                    order=interp_order,
                    prefilter=interp_prefilter,
                )

        return new_cube_flat.reshape(cube.shape[:~2] + new_cube_flat.shape[~2:])

    def plot_residual(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
            other: typ.Optional['Distortion'] = None,
            inverse: bool = False,
            use_titles: bool = True,
            use_xlabels: bool = True,
    ) -> typ.MutableSequence[plt.Axes]:

        if other is None:
            other = self

        wavelength = other.wavelength
        mesh_input = other.spatial_mesh_input
        residual = self.residual(other, inverse=inverse)
        residual_mag = vector.length(residual, keepdims=False)

        if config_index is not None:
            wavelength = wavelength[config_index]
            mesh_input = mesh_input[config_index]
            residual_mag = residual_mag[config_index]

        grid_shape = np.broadcast(wavelength, mesh_input[x], residual_mag).shape
        vgrid_shape = grid_shape + mesh_input.shape[~0:]
        wavelength = np.broadcast_to(wavelength, grid_shape, subok=True)
        mesh_input = np.broadcast_to(mesh_input, vgrid_shape, subok=True)
        residual_mag = np.broadcast_to(residual_mag, grid_shape, subok=True)

        if axs is None:
            fig, axs = plt.subplots(ncols=len(wavelength))
        else:
            fig = axs[0].figure

        vmin, vmax = residual_mag.min(), residual_mag.max()

        model = self.model()
        for ax, wavl, mesh_in, res in zip(axs, wavelength, mesh_input, residual_mag):
            if use_titles:
                ax.set_title(fmt.quantity(wavl[0, 0]))

            min_x, min_y = mesh_in[x].min(), mesh_in[y].min()
            max_x, max_y = mesh_in[x].max(), mesh_in[y].max()
            if inverse:
                min_x, min_y = model(wavl[0, 0], min_x, min_y)[0, 0, 0]
                max_x, max_y = model(wavl[0, 0], max_x, max_y)[0, 0, 0]

            img = ax.imshow(
                X=res.value.T,
                vmin=vmin.value,
                vmax=vmax.value,
                origin='lower',
                extent=[min_x.value, max_x.value, min_y.value, max_y.value],
            )
            if use_xlabels:
                ax.set_xlabel('input $x$ ' + '(' + "{0:latex}".format(mesh_input.unit) + ')')

        axs[0].set_ylabel('input $y$ ' + '(' + "{0:latex}".format(mesh_input.unit) + ')')

        fig.colorbar(img, ax=axs, label='residual mag. (' + '{0:latex}'.format(residual_mag.unit) + ')')

        return axs

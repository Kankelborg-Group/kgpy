import typing as typ
import dataclasses
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.plot
import kgpy.function
from kgpy import mixin, vector, format as fmt

__all__ = [
    'Distortion',
    'Vignetting',
    'psf',
    'Aberration',
]


@dataclasses.dataclass
class Distortion:
    wavelength: u.Quantity
    spatial_mesh_input: vector.Vector2D
    spatial_mesh_output: vector.Vector2D
    mask: np.ndarray
    polynomial_degree: int = 1

    def model(self, inverse: bool = False) -> 'polynomial.Vector2DValuedPolynomial3D':
        if not inverse:
            mesh_input, mesh_output = self.spatial_mesh_input, self.spatial_mesh_output
            names_input, names_output = ['x', 'y', '\\lambda', ], ['$x\'$', '$y\'$']
        else:
            mesh_input, mesh_output = self.spatial_mesh_output, self.spatial_mesh_input
            names_input, names_output = ['x\'', 'y\'', '\\lambda', ], ['$x$', '$y$']
        mesh_input = mesh_input.to_3d(self.wavelength)
        model = polynomial.Vector2DValuedPolynomial3D.from_lstsq_fit(
            # x=self.wavelength,
            # y=mesh_input.x,
            # z=mesh_input.y,
            # data=mesh_output,
            data_input=mesh_input,
            data_output=mesh_output,
            mask=self.mask,
            degree=self.polynomial_degree,
            input_names=names_input,
            output_names=names_output,
        )
        # model.component_names_input = names_input
        # model.component_names_output = names_output
        return model

    @property
    def plate_scale(self) -> vector.Vector2D:
        model = self.model()
        # output_max = self.spatial_mesh_input.max()
        # output_min = self.spatial_mesh_input.min()
        # dy = model(output_max.to_3d(self.wavelength)) - model(output_min.to_3d(self.wavelength))
        # dx = output_max - output_min
        # return dx / dy
        center_fov = vector.Vector3D(x=0 * u.arcsec, y=0 * u.arcsec, z=self.wavelength)
        # return 1 / model.dx(center_fov)
        return vector.Vector2D(
            x=1 / model.dx(center_fov).length,
            y=1 / model.dy(center_fov).length,
        )

    @property
    def dispersion(self) -> u.Quantity:
        model = self.model()
        center_fov = vector.Vector3D(x=0 * u.arcsec, y=0 * u.arcsec, z=self.wavelength)
        return 1 / model.dz(center_fov).length

    def residual(
            self,
            other: typ.Optional['Distortion'] = None,
            inverse: bool = False,
    ) -> vector.Vector2D:
        if other is None:
            other = self
        mesh_input, mesh_output = other.spatial_mesh_input, other.spatial_mesh_output
        if inverse:
            mesh_input, mesh_output = mesh_output, mesh_input
        model = self.model(inverse=inverse)
        vector_input = mesh_input.to_3d(other.wavelength)
        # print(mesh_output.unit)
        # print(model(vector_input).unit)
        residual = mesh_output - model(vector_input)
        residual[~other.mask] = 0
        return residual

    def __call__(
            self,
            cube: u.Quantity,
            wavelength: u.Quantity,
            spatial_input_min: vector.Vector2D,
            spatial_input_max: vector.Vector2D,
            spatial_output_min: vector.Vector2D,
            spatial_output_max: vector.Vector2D,
            spatial_samples_output: typ.Union[int, vector.Vector2D],
            inverse: bool = False,
            # channel_index: typ.Optional[int] = None,
            interp_order: int = 1,
            interp_prefilter: bool = False,
            fill_value: float = np.nan,
    ) -> np.ndarray:

        # if isinstance(spatial_samples_output, int):
        #     spatial_samples_output = vector.Vector2D(spatial_samples_output, spatial_samples_output)

        output_grid = vector.Vector3D()
        output_grid.x = np.linspace(spatial_output_min.x, spatial_output_max.x, spatial_samples_output.x)
        output_grid.y = np.linspace(spatial_output_min.y, spatial_output_max.y, spatial_samples_output.y)
        output_grid.x = output_grid.x[..., :, np.newaxis, np.newaxis]
        output_grid.y = output_grid.y[..., np.newaxis, :, np.newaxis]
        wavelength = wavelength[..., np.newaxis, np.newaxis, :]
        output_grid.z = wavelength

        # output_grid_x = np.linspace(output_min[x], output_max[x], spatial_samples_output[x])
        # output_grid_y = np.linspace(output_min[y], output_max[y], spatial_samples_output[y])
        # wavelength, output_grid_x, output_grid_y = np.broadcast_arrays(
        #     wavelength[..., None, None], output_grid_x[..., None], output_grid_y, subok=True)

        model = self.model(inverse=not inverse)

        # coordinates = model(wavelength, output_grid_x, output_grid_y)
        coordinates = model(output_grid)
        coordinates = (coordinates - spatial_input_min) / (spatial_input_max - spatial_input_min)
        # coordinates *= cube.shape[~1:] * u.pix
        coordinates = coordinates * vector.Vector2D(x=cube.shape[~2] * u.pix, y=cube.shape[~1] * u.pix)
        coordinates = coordinates.to_3d(wavelength)

        sh = cube.shape[:~2]
        coordinates = np.broadcast_to(coordinates, sh + coordinates.shape[~2:])

        coordinates_flat = coordinates.reshape((-1, ) + coordinates.shape[~2:])
        # coordinates_flat = np.moveaxis(coordinates_flat, ~0, 2)

        cube_flat = cube.reshape((-1,) + cube.shape[~2:])

        new_cube_flat_shape = list(cube_flat.shape)
        new_cube_flat_shape[~2] = spatial_samples_output.x
        new_cube_flat_shape[~1] = spatial_samples_output.y
        new_cube_flat = np.empty(new_cube_flat_shape)

        for i in range(cube_flat.shape[0]):
            # for j in range(cube_flat.shape[~0]):
            coords = coordinates_flat[i]
            new_cube_flat[i] = scipy.ndimage.map_coordinates(
                input=cube_flat[i],
                coordinates=np.stack([coords.x, coords.y, coords.z]),
                order=interp_order,
                prefilter=interp_prefilter,
                cval=fill_value,
            )

        return new_cube_flat.reshape(cube.shape[:~2] + new_cube_flat.shape[~2:]) << cube.unit

    def plot_residual(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
            other: typ.Optional['Distortion'] = None,
            inverse: bool = False,
            use_titles: bool = True,
            use_xlabels: bool = True,
            wavelength_name: typ.Optional[np.ndarray] = None,
    ) -> typ.MutableSequence[plt.Axes]:

        if other is None:
            other = self

        wavelength = other.wavelength
        # sorted_indices = np.argsort(wavelength[0, 0])
        wavelength = wavelength
        mesh_input = other.spatial_mesh_input
        residual = self.residual(other, inverse=inverse)
        residual_mag = residual.length
        # residual_mag = vector.length(residual, keepdims=False)

        if config_index is not None:
            wavelength = wavelength[config_index]
            mesh_input = mesh_input[config_index]
            residual_mag = residual_mag[config_index]

        grid_shape = np.broadcast(wavelength, mesh_input, residual_mag).shape
        # vgrid_shape = grid_shape + mesh_input.shape[~0:]
        # wavelength = np.broadcast_to(wavelength, grid_shape, subok=True)
        wavelength = wavelength.squeeze()
        mesh_input = np.broadcast_to(mesh_input, grid_shape, subok=True)
        residual_mag = np.broadcast_to(residual_mag, grid_shape, subok=True)

        if axs is None:
            fig, axs = plt.subplots(ncols=len(wavelength))
        else:
            fig = axs[0].figure

        if wavelength_name is None:
            wavelength_name = wavelength.copy()

        wsl = slice(None, len(axs))
        wavelength = wavelength[..., wsl]
        residual_mag = residual_mag[..., wsl]

        sorted_indices = np.argsort(wavelength)
        wavelength = wavelength[..., sorted_indices]
        wavelength_name = wavelength_name[..., sorted_indices]
        residual_mag = residual_mag[..., sorted_indices]

        vmin, vmax = residual_mag.min(), residual_mag.max()

        model = self.model()
        # for ax, wavl, mesh_in, res in zip(axs, wavelength, mesh_input, residual_mag):
        for i in range(len(axs)):
            wavl = wavelength[i]
            if use_titles:
                axs[i].set_title(wavelength_name[i])

            mesh_in = mesh_input[..., i]
            min_x, min_y = mesh_in.x.min(), mesh_in.y.min()
            max_x, max_y = mesh_in.x.max(), mesh_in.y.max()
            if inverse:
                min_x, min_y = model(wavl, min_x, min_y)[0, 0, 0]
                max_x, max_y = model(wavl, max_x, max_y)[0, 0, 0]

            img = axs[i].imshow(
                X=residual_mag[..., i].value.T,
                vmin=vmin.value,
                vmax=vmax.value,
                origin='lower',
                extent=[min_x.value, max_x.value, min_y.value, max_y.value],
            )
            if use_xlabels:
                axs[i].set_xlabel('input $x$ ' + '(' + "{0:latex}".format(mesh_input.x.unit) + ')')

        axs[0].set_ylabel('input $y$ ' + '(' + "{0:latex}".format(mesh_input.y.unit) + ')')

        fig.colorbar(img, ax=axs, label='residual mag. (' + '{0:latex}'.format(residual_mag.unit) + ')')

        return axs


@dataclasses.dataclass
class Vignetting:
    wavelength: u.Quantity
    spatial_mesh: vector.Vector2D
    unvignetted_percent: u.Quantity
    mask: np.ndarray
    polynomial_degree: int = 1

    @property
    def mesh(self) -> vector.Vector3D:
        return self.spatial_mesh.to_3d(z=self.wavelength)

    def __call__(
            self,
            cube: np.ndarray,
            wavelength: u.Quantity,
            # spatial_domain: u.Quantity,
            spatial_min: vector.Vector2D,
            spatial_max: vector.Vector2D,
            inverse: bool = False,
    ) -> np.ndarray:
        return Vignetting.apply_model(
            model=self.model(inverse=inverse),
            cube=cube,
            wavelength=wavelength,
            # spatial_domain=spatial_domain,
            spatial_min=spatial_min,
            spatial_max=spatial_max,
        )

    @staticmethod
    def apply_model(
            model: 'polynomial.Polynomial3D',
            cube: np.ndarray,
            wavelength: u.Quantity,
            spatial_min: vector.Vector2D,
            spatial_max: vector.Vector2D,
            # spatial_domain: u.Quantity,
    ):

        # grid_x = np.linspace(output_min[x], output_max[x], cube.shape[~1])
        # grid_y = np.linspace(output_min[y], output_max[y], cube.shape[~0])
        # wavelength, grid_x, grid_y = np.broadcast_arrays(wavelength[..., None, None], grid_x[..., None], grid_y, subok=True)
        grid = vector.Vector3D()
        grid.x = np.linspace(spatial_min.x, spatial_max.x, cube.shape[~2])[..., :, np.newaxis, np.newaxis]
        grid.y = np.linspace(spatial_min.y, spatial_max.y, cube.shape[~1])[..., np.newaxis, :, np.newaxis]
        grid.z = wavelength[..., np.newaxis, np.newaxis, :]

        # vig = model(wavelength, grid_x, grid_y).to(u.dimensionless_unscaled)[..., 0]
        vig = model(grid)

        return vig * cube

    def model(self, inverse: bool = False) -> 'polynomial.Polynomial3D':
        data = self.unvignetted_percent
        names_input, names_output = ['x', 'y', '\\lambda', ], '$V$'
        if inverse:
            data = 1 / data
            names_output = '$V\'$'

        model = polynomial.Polynomial3D.from_lstsq_fit(
            # x=self.wavelength,
            # y=self.spatial_mesh[x],
            # z=self.spatial_mesh[y],
            # data=data[..., None],
            data_input=self.mesh,
            data_output=data,
            mask=self.mask,
            degree=self.polynomial_degree,
            input_names=names_input,
            output_name=names_output,
        )
        # model.input_names = names_input
        # model.output_name = names_output
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
        # residual = data[..., None] - model(x=other.wavelength, y=other.spatial_mesh[x], z=other.spatial_mesh[y])
        # residual = residual[..., 0]
        residual = data - model(vector_input=other.mesh)
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
            wavelength_name: typ.Optional[np.ndarray] = None,
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
            wavelength_name=wavelength_name,
        )

    def plot_unvignetted(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
            other: typ.Optional['Vignetting'] = None,
            inverse: bool = False,
            use_titles: bool = True,
            use_xlabels: bool = True,
            wavelength_name: typ.Optional[np.ndarray] = None,
    ) -> typ.MutableSequence[plt.Axes]:
        if other is None:
            other = self
        data = self.unvignetted_percent
        data[data == 0] = np.nan
        if inverse:
            data = 1 / data
        return Vignetting.plot(
            wavelength=other.wavelength,
            spatial_mesh=other.spatial_mesh,
            data=data,
            axs=axs,
            config_index=config_index,
            data_name='effective area',
            use_titles=use_titles,
            use_xlabels=use_xlabels,
            wavelength_name=wavelength_name,
        )

    @staticmethod
    def plot(
            wavelength: u.Quantity,
            spatial_mesh: vector.Vector2D,
            data: u.Quantity,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            config_index: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None,
            data_name: str = '',
            use_titles: bool = True,
            use_xlabels: bool = True,
            wavelength_name: typ.Optional[np.ndarray] = None,
    ) -> typ.MutableSequence[plt.Axes]:

        if config_index is not None:
            wavelength = wavelength[config_index]
            spatial_mesh = spatial_mesh[config_index]
            data = data[config_index]

        grid_shape = np.broadcast(wavelength, spatial_mesh, data).shape
        # vgrid_shape = grid_shape + spatial_mesh.shape[~0:]
        # wavelength = np.broadcast_to(wavelength, grid_shape, subok=True)
        wavelength = wavelength[0, 0]
        spatial_mesh = np.broadcast_to(spatial_mesh, grid_shape, subok=True)
        data = np.broadcast_to(data, grid_shape, subok=True)

        if wavelength_name is None:
            wavelength_name = wavelength

        if axs is None:
            fig, axs = plt.subplots(ncols=len(wavelength))
        else:
            fig = axs[0].figure

        wsl = slice(None, len(axs))
        wavelength = wavelength[..., wsl]
        data = data[..., wsl]

        # vmin, vmax = data.min(), data.max()
        vmin, vmax = np.nanmin(data), np.nanmax(data)

        # for ax, wavl, mesh, d in zip(axs, wavelength, spatial_mesh, data):
        for i in range(len(axs)):
            if use_titles:
                axs[i].set_title(wavelength_name[i])

            mesh = spatial_mesh[..., i]

            extent = kgpy.plot.calc_extent(
                data_min=mesh.min(),
                data_max=mesh.max(),
                num_steps=kgpy.vector.Vector2D.from_quantity(mesh.shape * u.dimensionless_unscaled),
            )

            img = axs[i].imshow(
                X=data[..., i].value.T,
                vmin=vmin.value,
                vmax=vmax.value,
                origin='lower',
                extent=[e.value for e in extent],
            )
            if use_xlabels:
                axs[i].set_xlabel('input $x$ ' + '(' + "{0:latex}".format(spatial_mesh.x.unit) + ')')

        axs[0].set_ylabel('input $y$ ' + '(' + "{0:latex}".format(spatial_mesh.y.unit) + ')')

        fig.colorbar(img, ax=axs, label=data_name + ' (' + '{0:latex}'.format(data.unit) + ')')

        return axs


from . import psf


@dataclasses.dataclass
class Aberration:

    distortion: Distortion
    vignetting: Vignetting

    def __call__(
            self,
            data: u.Quantity,
            wavelength: u.Quantity,
            # spatial_domain_input: u.Quantity,
            # spatial_domain_output: u.Quantity,
            # spatial_samples_output: typ.Union[int, typ.Tuple[int, int]],
            spatial_input_min: vector.Vector2D,
            spatial_input_max: vector.Vector2D,
            spatial_output_min: vector.Vector2D,
            spatial_output_max: vector.Vector2D,
            spatial_samples_output: typ.Union[int, vector.Vector2D],
            inverse: bool = False,
    ) -> u.Quantity:
        if not inverse:
            data = self.vignetting(
                cube=data,
                wavelength=wavelength,
                # spatial_domain=spatial_domain_input,
                spatial_min=spatial_input_min,
                spatial_max=spatial_input_max,
                inverse=inverse,
            )
            data = self.distortion(
                cube=data,
                wavelength=wavelength,
                # spatial_domain_input=spatial_domain_input,
                # spatial_domain_output=spatial_domain_output,
                spatial_input_min=spatial_input_min,
                spatial_input_max=spatial_input_max,
                spatial_output_min=spatial_output_min,
                spatial_output_max=spatial_output_max,
                spatial_samples_output=spatial_samples_output,
                inverse=inverse,
                fill_value=np.nan,
            )
        else:
            data = self.distortion(
                cube=data,
                wavelength=wavelength,
                # spatial_domain_input=spatial_domain_input,
                # spatial_domain_output=spatial_domain_output,
                spatial_input_min=spatial_input_min,
                spatial_input_max=spatial_input_max,
                spatial_output_min=spatial_output_min,
                spatial_output_max=spatial_output_max,
                spatial_samples_output=spatial_samples_output,
                inverse=inverse,
                fill_value=np.nan,
            )
            data = self.vignetting(
                cube=data,
                wavelength=wavelength,
                # spatial_domain=spatial_domain_output,
                spatial_min=spatial_output_min,
                spatial_max=spatial_output_max,
                inverse=inverse,
            )
        return data

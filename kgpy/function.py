import typing as typ
import abc
import dataclasses
import copy
import numpy as np
import numpy.typing
import matplotlib.axes
import scipy.interpolate
import astropy.units as u
import astropy.visualization
import numba
import kgpy.mixin
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.matrix

__all__ = [

]

InputT = typ.TypeVar('InputT', bound=kgpy.vectors.VectorLike)
OutputT = typ.TypeVar('OutputT', bound=kgpy.vectors.VectorLike)
AbstractArrayT = typ.TypeVar('AbstractArrayT', bound='AbstractArray')
ArrayT = typ.TypeVar('ArrayT', bound='Array')
PolynomialArrayT = typ.TypeVar('PolynomialArrayT', bound='PolynomialArray')


@dataclasses.dataclass(eq=False)
class AbstractArray(
    kgpy.mixin.Copyable,
    abc.ABC,
    typ.Generic[InputT, OutputT],
):

    input: InputT

    @property
    def input_broadcasted(self: InputT) -> InputT:
        return np.broadcast_to(self.input, self.shape)

    @property
    @abc.abstractmethod
    def output(self: AbstractArrayT) -> OutputT:
        pass

    @property
    @abc.abstractmethod
    def mask(self: AbstractArrayT) -> kgpy.uncertainty.ArrayLike:
        return

    @property
    def shape(self: AbstractArrayT) -> typ.Dict[str, int]:
        return kgpy.labeled.Array.broadcast_shapes(self.input, self.output)

    @abc.abstractmethod
    def __call__(self: AbstractArrayT, input_new: InputT) -> OutputT:
        pass

    @property
    def inverse(self) -> ArrayT:
        return Array(
            input=self.output,
            output=self.input,
        )

    def pcolormesh(
            self: AbstractArrayT,
            axs: numpy.typing.NDArray[matplotlib.axes.Axes],
            input_component_x: str,
            input_component_y: str,
            input_component_row: typ.Optional[str] = None,
            input_component_column: typ.Optional[str] = None,
            output_component_color: typ.Optional[str] = None,
            index: typ.Optional[typ.Dict[str, int]] = None,
            **kwargs,
    ):
        """
        Plot this function using :class:`matplotlib.pyplot.pcolormesh`.

        Examples
        --------

        .. jupyter-execute::

            import numpy as np
            import matplotlib.pyplot as plt
            import astropy.units as u
            import kgpy.labeled
            import kgpy.vectors
            import kgpy.matrix
            import kgpy.function

            a = kgpy.labeled.LinearSpace(0 * u.rad, np.pi * u.rad, num=2, axis='a')
            b = kgpy.labeled.LinearSpace(0 * u.rad, np.pi * u.rad, num=2, axis='b')
            x = kgpy.labeled.LinearSpace(0 * u.rad, 2 * np.pi * u.rad, num=11, axis='x')
            y = kgpy.labeled.LinearSpace(0 * u.rad, 1 * np.pi * u.rad, num=11, axis='y')

            x2 = 2 * x + y / 4
            y2 = x / 5 + y

            inputs = kgpy.vectors.CartesianND(dict(
                a=a,
                b=b,
                x=x2,
                y=y2,
            ))

            f = kgpy.function.Array(
                input=inputs,
                output=np.cos(a) * np.cos(b) * np.cos(x2) * np.cos(y2),
            )

            fig, axs = plt.subplots(
                nrows=a.num,
                ncols=b.num,
                sharex=True,
                sharey=True,
                squeeze=False,
                figsize=(12, 12),
                constrained_layout=True,
            )

            f.pcolormesh(
                axs=axs,
                input_component_x='x',
                input_component_y='y',
                input_component_row='a',
                input_component_column='b',
            )

        Parameters
        ----------
        axs: Array of axes that the function will be plotted on
        input_component_x
        input_component_y
        input_component_row
        input_component_column
        output_component_color
        index
        kwargs

        Returns
        -------

        """

        axs = kgpy.labeled.Array(axs, ['row', 'column'])

        if index is None:
            index = dict()

        with astropy.visualization.quantity_support():
            for index_subplot in axs.ndindex():

                index_final = index.copy()
                if input_component_row is not None:
                    index_final[input_component_row] = index_subplot['row']
                if input_component_column is not None:
                    index_final[input_component_column] = index_subplot['column']

                inp = self.input.broadcasted[index_final]
                inp_x = inp.coordinates_flat[input_component_x].array
                inp_y = inp.coordinates_flat[input_component_y].array

                out = self.output.broadcasted[index_final]
                if output_component_color is not None:
                    out = out.coordinates_flat[output_component_color]
                out = out.array

                ax = axs[index_subplot].array
                ax.pcolormesh(
                    inp_x,
                    inp_y,
                    out,
                    shading='nearest',
                    **kwargs,
                )

                if index_subplot['row'] == axs.shape['row'] - 1:
                    if isinstance(inp_x, u.Quantity):
                        ax.set_xlabel(f'{input_component_x} ({inp_x.unit})')
                    else:
                        ax.set_xlabel(f'{input_component_x}')
                else:
                    ax.set_xlabel(None)

                if index_subplot['column'] == 0:
                    if isinstance(inp_y, u.Quantity):
                        ax.set_ylabel(f'{input_component_y} ({inp_y.unit})')
                    else:
                        ax.set_ylabel(f'{input_component_y}')
                else:
                    ax.set_ylabel(None)

                if input_component_row is not None:
                    if index_subplot['row'] == 0:
                        inp_column = inp.coordinates_flat[input_component_column]
                        ax.text(
                            x=0.5,
                            y=1.01,
                            s=f'{input_component_column} = {inp_column.mean().array.value:0.03f} {inp_column.unit:latex_inline}',
                            transform=ax.transAxes,
                            ha='center',
                            va='bottom'
                        )

                if input_component_column is not None:
                    if index_subplot['column'] == axs.shape['column'] - 1:
                        inp_row = inp.coordinates_flat[input_component_row]
                        ax.text(
                            x=1.01,
                            y=0.5,
                            s=f'{input_component_row} = {inp_row.mean().array.value:0.03f} {inp_row.unit:latex_inline}',
                            transform=ax.transAxes,
                            va='center',
                            ha='left',
                            rotation=-90,
                        )


@dataclasses.dataclass
class Array(
    AbstractArray[InputT, OutputT],
):

    output: OutputT = None
    mask: typ.Optional[kgpy.uncertainty.ArrayLike] = None

    @property
    def output_broadcasted(self: ArrayT) -> OutputT:
        output = self.output
        if not isinstance(output, kgpy.labeled.ArrayInterface):
            output = kgpy.labeled.Array(output)
        return np.broadcast_to(output, shape=self.shape)

    def calc_index_nearest(self: ArrayT, input_new: InputT, ) -> typ.Dict[str, kgpy.labeled.Array]:

        if not isinstance(input_new, kgpy.vectors.AbstractVector):
            input_new = kgpy.vectors.Cartesian1D(input_new)

        input_old = self.input
        if not isinstance(input_old, kgpy.vectors.AbstractVector):
            input_old = kgpy.vectors.Cartesian1D(input_old)
        else:
            input_old = input_old.copy_shallow()

        for component in input_old.coordinates:
            coordinate = input_old.coordinates[component]
            coordinate = kgpy.labeled.Array(coordinate.array, coordinate.axes)
            coordinate.axes = [f'{ax}_dummy' for ax in coordinate.axes]
            setattr(input_old, component, coordinate)
        shape_dummy = input_old.shape

        distance = (input_new - input_old).length
        distance = distance.combine_axes(axes=shape_dummy.keys(), axis_new='dummy')

        index = np.argmin(distance, axis='dummy')
        index = np.unravel_index(index, self.input.shape)
        return index

    def interp_nearest(self: ArrayT, input_new: InputT) -> ArrayT:
        return type(self)(
            input=input_new,
            output=self.output_broadcasted[self.calc_index_nearest(input_new)]
        )

    def calc_index_lower(self: ArrayT, input_new: InputT ) -> typ.Dict:

        input_old = self.input
        if isinstance(input_old, kgpy.vectors.AbstractVector):
            input_old = input_old.copy_shallow()
        else:
            input_old = kgpy.vectors.Cartesian1D(input_old)

        for component in input_old.coordinates:
            coordinate = input_old.coordinates[component]
            coordinate = kgpy.labeled.Array(coordinate.array, coordinate.axes)
            coordinate.axes = [f'{ax}_dummy' for ax in coordinate.axes]
            setattr(input_old, component, coordinate)
        shape_dummy = input_old.shape

        distance = input_old - input_new
        distance[distance > 0] = -np.inf
        distance = distance.component_sum
        distance = distance.combine_axes(axes=shape_dummy.keys(), axis_new='dummy')

        index = np.argmax(distance, axis='dummy')
        index = np.unravel_index(index, self.input.shape)

        for axis in index:
            index_max = self.input.shape[axis] - 2
            index[axis][index[axis] > index_max] = index_max

        return index

    def _interp_linear_1d_recursive(
            self,
            input_new: InputT,
            index_lower: typ.Dict[str, kgpy.labeled.AbstractArray],
            axis_stack: typ.List[str],
    ) -> kgpy.labeled.AbstractArray:

        axis = axis_stack.pop(0)

        index_upper = copy.deepcopy(index_lower)
        index_upper[axis] = index_upper[axis] + 1

        x = input_new.coordinates[axis]

        input_old = self.input
        input_old = np.broadcast_to(input_old, input_old.shape)
        if isinstance(input_old, kgpy.vectors.AbstractVector):
            x0 = input_old.coordinates[axis][index_lower]
            x1 = input_old.coordinates[axis][index_upper]
        else:
            x0 = input_old[index_lower]
            x1 = input_old[index_upper]

        if axis_stack:
            y0 = self._interp_linear_1d_recursive(input_new=input_new, index_lower=index_lower, axis_stack=axis_stack.copy())
            y1 = self._interp_linear_1d_recursive(input_new=input_new, index_lower=index_upper, axis_stack=axis_stack.copy())

        else:
            output = self.output_broadcasted
            y0 = output[index_lower]
            y1 = output[index_upper]

        result = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

        return result

    def interp_linear(self: ArrayT, input_new: InputT, ) -> OutputT:
        if not isinstance(input_new, kgpy.vectors.AbstractVector):
            input_new = kgpy.vectors.Cartesian1D(input_new)
        return self._interp_linear_1d_recursive(
            input_new=input_new,
            index_lower=self.calc_index_lower(input_new),
            axis_stack=list(input_new.shape.keys()),
        )

    @staticmethod
    @numba.njit(parallel=True, )
    def _calc_index_nearest_even_numba(
            input_new: np.ndarray,
            input_old: np.ndarray,
            mask_input_old: np.ndarray,
    ) -> np.ndarray:

        shape_input_old = input_old.shape
        shape_input_new = input_new.shape

        index_nearest_even = np.empty(shape_input_new[1:], dtype=numba.int64)

        for index_input_new in numba.prange(shape_input_new[~0]):
            distance_squared_min = np.inf
            for index_input_old in numba.prange(shape_input_old[~0]):
                if mask_input_old[index_input_old]:
                    distance_squared = 0
                    for component in numba.prange(shape_input_new[0]):
                        dist = input_new[component, index_input_new] - input_old[component, index_input_old]
                        distance_squared += dist * dist

                    if distance_squared < distance_squared_min:
                        index_nearest_even[index_input_new] = index_input_old
                        distance_squared_min = distance_squared

        return index_nearest_even

    def _calc_index_nearest_even(
            self: ArrayT,
            input_new: InputT,
    ) -> typ.Dict[str, kgpy.uncertainty.ArrayLike]:

        input = self.input

        shape_input = input.shape
        input = input.reshape(dict(dummy=-1))

        mask = np.indices(shape_input.values()).sum(0) % 2 == 0
        mask = mask.reshape(-1)

        shape_input_new = input_new.shape
        input_new = input_new.broadcasted.reshape(dict(dummy=-1))

        index = Array._calc_index_nearest_even_numba(
            input_new=input_new.array,
            input_old=input.array,
            mask_input_old=mask,
        )

        index = kgpy.labeled.Array(index, axes=['dummy'])
        index = index.reshape(shape_input_new)
        index = np.unravel_index(index, shape=shape_input)

        return index

    def interp_barycentric_linear(
            self: ArrayT,
            input_new: InputT,
    ) -> ArrayT:
        """
        Interpolate this function using barycentric interpolation.

        Examples
        --------

        .. jupyter-execute::

            import numpy as np
            import matplotlib.pyplot as plt
            import astropy.units as u
            import kgpy.labeled
            import kgpy.vectors
            import kgpy.matrix
            import kgpy.function

            input = kgpy.vectors.Cartesian2D(
                x=kgpy.labeled.LinearSpace(-np.pi, np.pi, num=14, axis='x'),
                y=kgpy.labeled.LinearSpace(-np.pi, np.pi, num=14, axis='y'),
            )

            input_rotated = kgpy.matrix.Cartesian2D.rotation(30 * u.deg) @ input

            f = kgpy.function.Array(
                input=input_rotated,
                output=np.cos(input.x * input.x * u.rad) * np.cos(input.y * input.y * u.rad),
            )

            fig, axs = plt.subplots(squeeze=False)
            f.pcolormesh(
                axs=axs,
                input_component_x='x',
                input_component_y='y',
            )

        .. jupyter-execute::

            input_large = kgpy.vectors.Cartesian2D(
                x=kgpy.labeled.LinearSpace(-2 * np.pi, 2 * np.pi, num=201, axis='x'),
                y=kgpy.labeled.LinearSpace(-2 * np.pi, 2 * np.pi, num=201, axis='y'),
            )

            g = f.interp_barycentric_linear(input_large)

            fig_large, axs_large = plt.subplots(squeeze=False)
            g.pcolormesh(
                axs=axs_large,
                input_component_x='x',
                input_component_y='y',
            )
            axs_large[0, 0].set_xlim(axs[0, 0].get_xlim());
            axs_large[0, 0].set_ylim(axs[0, 0].get_ylim());


        Parameters
        ----------
        input_new

        Returns
        -------
        A new instance of class:`kgpy.function.Array` evaluated at the coordinates `input_new`.
        """

        index_nearest = self._calc_index_nearest_even(input_new)

        num_vertices = len(input_new.coordinates) + 1
        index = dict()
        axes_simplex = []
        for a, axis in enumerate(index_nearest.keys()):
            axis_simplex = f'simplex_{axis}'
            axes_simplex.append(axis_simplex)
            simplex_axis = kgpy.labeled.Array.zeros(shape={axis_simplex: 2, 'vertices': num_vertices}, dtype=int)
            simplex_axis[dict(vertices=a+1)] = kgpy.labeled.Array(np.array([-1, 1], dtype=int), axes=[axis_simplex])
            index[axis] = index_nearest[axis] + simplex_axis

        shape_index = kgpy.labeled.Array.broadcast_shapes(*index.values())
        barycentric_transform_shape = shape_index.copy()
        barycentric_transform_shape['vertices'] = len(input_new.coordinates)
        barycentric_transform_shape['component'] = len(input_new.coordinates)
        barycentric_transform = kgpy.labeled.Array.empty(barycentric_transform_shape)
        if isinstance(input_new.unit, u.UnitBase):
            barycentric_transform = barycentric_transform << input_new.unit

        index_0 = {k: index[k][dict(vertices=0)] for k in index}
        index_1 = {k: index[k][dict(vertices=slice(1, None))] % self.input.shape[k] for k in index}

        for c, component in enumerate(self.input.coordinates):
            x0 = self.input.coordinates[component][index_0]
            x1 = self.input.coordinates[component][index_1]
            barycentric_transform[dict(component=c)] = x1 - x0

        barycentric_transform = barycentric_transform.matrix_inverse(
            axis_rows='component',
            axis_columns='vertices',
        )

        barycentric_coordinates = input_new - self.input[index_nearest]
        barycentric_coordinates = barycentric_coordinates.array_labeled
        barycentric_coordinates = barycentric_coordinates.add_axes(axes_simplex + ['vertices'])

        barycentric_coordinates = barycentric_transform.matrix_multiply(
            barycentric_coordinates,
            axis_rows='component',
            axis_columns='vertices',
        ).combine_axes(['vertices', 'component'], axis_new='vertices')

        epsilon = 1e-15
        mask_inside = (-epsilon <= barycentric_coordinates) & (barycentric_coordinates <= 1 + epsilon)
        for axis in index:
            mask_inside = mask_inside & (index[axis][dict(vertices=slice(1, None))] >= 0)
            mask_inside = mask_inside & (index[axis][dict(vertices=slice(1, None))] < self.input.shape[axis])
        mask_inside = np.all(mask_inside, axis='vertices')

        shape_weights = shape_index
        weights = kgpy.labeled.Array.empty(shape_weights)
        weights[dict(vertices=0)] = 1 - np.sum(barycentric_coordinates, axis='vertices')
        weights[dict(vertices=slice(1, None))] = barycentric_coordinates

        output_new = weights * self.output_broadcasted[{k: index[k] % self.input.shape[k] for k in index}]
        output_new = np.nansum(output_new, axis='vertices')

        mask_inside = np.broadcast_to(mask_inside, shape=output_new.shape, subok=True)

        return Array(
            input=input_new,
            output=np.mean(output_new, axis=axes_simplex, where=mask_inside),
        )

    def interp_barycentric_linear_scipy(self: ArrayT, grid):

        axes_uninterpolated = self.grid.keys() - grid.keys()
        print('axes_uninterpolated', axes_uninterpolated)

        # shape_grid = kgpy.labeled.Array.broadcast_shapes(*grid.values())
        grid = grid.broadcasted

        value_interp = scipy.interpolate.griddata(
            points=tuple(val.data.reshape(-1) for val in self.grid.broadcasted.coordinates),
            values=self.value_broadcasted.data.reshape(-1),
            xi=tuple(val.data.reshape(-1) for val in grid.values()),
        )

        value_interp = kgpy.labeled.Array(
            value=value_interp.reshape(tuple(grid.shape.values())),
            axes=list(grid.keys()),
        )

        return Array(
            value=value_interp,
            grid=grid,
        )

    def __call__(
            self: ArrayT,
            grid,
    ) -> ArrayT:
        return self.interp_barycentric_linear(grid=grid)


@dataclasses.dataclass
class AbstractPolynomial(
    AbstractArray,
):
    @property
    @abc.abstractmethod
    def coefficients(self: PolynomialArrayT) -> kgpy.vectors.AbstractVector:
        pass

    @property
    def inverse(self) -> ArrayT:
        return PolynomialArray(
            input=self.output,
            output=self.input,
        )

    def _design_matrix_recursive(
            self: PolynomialArrayT,
            result: typ.Dict[str, kgpy.uncertainty.ArrayLike],
            coordinates: typ.Dict[str, kgpy.uncertainty.ArrayLike],
            key: str,
            value: kgpy.uncertainty.ArrayLike,
            degree_current: int,
            degree: int,
    ) -> None:
        component = next(iter(coordinates))
        coordinate = coordinates.pop(component)
        for i in range(degree + 1):
            if i > 0:
                if key:
                    key = f'{key},{component}'
                else:
                    key = component
                value = value * coordinate
                degree_current = degree_current + 1
            else:
                degree_current = degree_current

            if coordinates:
                self._design_matrix_recursive(result, coordinates.copy(), key, value, degree_current, degree)
            elif degree_current == degree:
                result[key] = value

    def design_matrix(self: PolynomialArrayT, inp: InputT) -> kgpy.vectors.CartesianND:
        result = dict()
        for d in range(self.degree + 1):
            self._design_matrix_recursive(result, inp.coordinates.copy(), key='', value=1, degree_current=0, degree=d)
        return kgpy.vectors.CartesianND(result)

    def __call__(self: PolynomialArrayT, input: InputT):
        coefficients = self.coefficients
        design_matrix = self.design_matrix(input)
        result = (coefficients * design_matrix).component_sum
        return result


@dataclasses.dataclass
class Polynomial(
    AbstractPolynomial,
):
    coefficients: kgpy.vectors.AbstractVector

    @property
    def output(self):
        return


@dataclasses.dataclass
class PolynomialArray(
    AbstractPolynomial,
    Array[InputT, OutputT],
):

    degree: int = 1
    axes_model: typ.Optional[typ.Union[str, typ.List[str]]] = None

    def __post_init__(self: PolynomialArrayT) -> None:
        self.update()

    def update(self: PolynomialArrayT) -> None:
        self._coefficients = None

    @property
    def coefficients(self: PolynomialArrayT) -> kgpy.vectors.AbstractVector:
        if self._coefficients is None:
            self._coefficients = self._calc_coefficients()
        return self._coefficients

    def _calc_coefficients(self: PolynomialArrayT) -> kgpy.vectors.AbstractVector:
        inp = self.input
        if not isinstance(inp, kgpy.vectors.AbstractVector):
            inp = kgpy.vectors.Cartesian1D(inp)

        mask = self.mask

        design_matrix = self.design_matrix(inp)
        design_matrix = np.broadcast_to(design_matrix, design_matrix.shape)

        gram_matrix = kgpy.matrix.CartesianND()
        for component_row in design_matrix.coordinates:
            gram_matrix.coordinates[component_row] = kgpy.vectors.CartesianND()
            for component_column in design_matrix.coordinates:
                element_row = design_matrix.coordinates[component_row]
                element_column = design_matrix.coordinates[component_column]
                element = element_row * element_column
                if mask is not None:
                    element[~mask] = 0 * element[~mask]
                element = np.sum(element, axis=self.axes_model)
                gram_matrix.coordinates[component_row].coordinates[component_column] = element

        gram_matrix_inverse = gram_matrix.inverse_numpy()

        output = self.output
        if not isinstance(output, kgpy.vectors.AbstractVector):
            output = kgpy.vectors.Cartesian1D(output)

        moment_matrix = kgpy.matrix.CartesianND()
        for component_row in design_matrix.coordinates:
            moment_matrix.coordinates[component_row] = type(output)()
            for component_column in output.coordinates:
                element_row = design_matrix.coordinates[component_row]
                element_column = output.coordinates[component_column]
                element = element_row * element_column
                if mask is not None:
                    element[~mask] = 0 * element[~mask]
                element = np.sum(element, axis=self.axes_model)
                moment_matrix.coordinates[component_row].coordinates[component_column] = element

        result = gram_matrix_inverse @ moment_matrix

        result = result.transpose

        if not isinstance(self.output, kgpy.vectors.AbstractVector):
            result = result.x
        else:
            result = result.transpose

        return result

    @property
    def residual(self: PolynomialArrayT) -> OutputT:
        result = self.output - self(self.input)
        result[~self.mask] = np.nan
        return result

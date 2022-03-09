import typing as typ
import abc
import dataclasses
import copy
import numpy as np
import numpy.typing
import matplotlib.axes
import scipy.interpolate
import astropy.visualization
import numba
import kgpy.mixin
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vector

__all__ = [

]

InputT = typ.TypeVar('InputT', bound=kgpy.vector.VectorLike)
OutputT = typ.TypeVar('OutputT', bound=kgpy.vector.VectorLike)
AbstractArrayT = typ.TypeVar('AbstractArrayT', bound='AbstractArray')
ArrayT = typ.TypeVar('ArrayT', bound='Array')


@dataclasses.dataclass(eq=False)
class AbstractArray(
    kgpy.mixin.Copyable,
    abc.ABC,
    typ.Generic[InputT, OutputT],
):

    input: InputT

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
        axs = kgpy.labeled.Array(axs, ['row', 'column'])

        if index is None:
            index = dict()

        with astropy.visualization.quantity_support():
            for index_subplot in axs.ndindex():

                print(index_subplot)

                index_final = {
                    **index,
                    input_component_row: index_subplot['row'],
                    input_component_column: index_subplot['column'],
                }

                inp = self.input.broadcasted[index_final]
                inp_x = inp.coordinates_flat[input_component_x].array
                inp_y = inp.coordinates_flat[input_component_y].array
                inp_row = inp.coordinates_flat[input_component_row]
                inp_column = inp.coordinates_flat[input_component_column]

                print(inp_column)

                out = self.output.broadcasted[index_final]
                if output_component_color is not None:
                    out = out.coordinates_flat[output_component_color]
                out = out.array

                ax = axs[index_subplot].array
                ax.pcolormesh(
                    inp_x,
                    inp_y,
                    out,
                    **kwargs,
                )

                if index_subplot['row'] == 0:
                    ax.set_xlabel(inp_x.unit)
                elif index_subplot['row'] == axs.shape['row'] - 1:
                    ax.set_xlabel(f'{inp_column.mean().array.value:0.03f} {inp_column.unit:latex_inline}')


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

        if not isinstance(input_new, kgpy.vector.AbstractVector):
            input_new = kgpy.vector.Cartesian1D(input_new)

        input_old = self.input
        if not isinstance(input_old, kgpy.vector.AbstractVector):
            input_old = kgpy.vector.Cartesian1D(input_old)
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
        if isinstance(input_old, kgpy.vector.AbstractVector):
            input_old = input_old.copy_shallow()
        else:
            input_old = kgpy.vector.Cartesian1D(input_old)

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

        # grid_data = self.grid_normalized
        #
        # shape_dummy = dict()
        # for axis_name_data in grid_data:
        #     axis_names = grid_data[axis_name_data].axis_names
        #     for axis_name in grid:
        #         if axis_name in axis_names:
        #             axis_name_dummy = f'{axis_name}_dummy'
        #             shape_dummy[axis_name_dummy] = self.shape[axis_name]
        #             axis_index = axis_names.index(axis_name)
        #             axis_names[axis_index] = axis_name_dummy
        #
        # distance_squared = 0
        # for axis_name in grid:
        #     d = grid_data[axis_name] - grid[axis_name]
        #     d[d > 0] = -np.inf
        #     distance_squared = distance_squared + d
        #
        # distance_squared = distance_squared.combine_axes(axes=shape_dummy.keys(), axis_new='dummy')
        # index = np.argmax(distance_squared, axis='dummy')
        # index = np.unravel_index(index, shape_dummy)
        #
        # for axis in index:
        #     index_max = shape_dummy[axis] - 2
        #     index[axis][index[axis] > index_max] = index_max
        #
        # index = {k[:~(len('_dummy') - 1)]: index[k] for k in index}
        #
        # return index

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
        if isinstance(input_old, kgpy.vector.AbstractVector):
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
        if not isinstance(input_new, kgpy.vector.AbstractVector):
            input_new = kgpy.vector.Cartesian1D(input_new)
        return self._interp_linear_1d_recursive(
            input_new=input_new,
            index_lower=self.calc_index_lower(input_new),
            axis_stack=list(input_new.shape.keys()),
        )

    # def interp_idw(
    #         self,
    #         grid: typ.Dict[str, LabeledArray],
    #         power: float = 2,
    # ) -> 'DataArray':
    #
    #     grid_data = self.grid_broadcasted
    #
    #     index = self.calc_index_lower(**grid)
    #
    #     axes_kernel = []
    #     for axis in grid:
    #         axis_kernel = f'kernel_{axis}'
    #         axes_kernel.append(axis_kernel)
    #         index[axis] = index[axis] + LabeledArray.arange(stop=2, axis=axis_kernel)
    #
    #     distance_to_power = 0
    #     for axis in grid:
    #         distance_to_power = distance_to_power + np.abs(grid_data[axis][index] - grid[axis]) ** power
    #     distance = distance_to_power ** (1 / power)
    #
    #     weights = 1 / distance
    #
    #     return DataArray(
    #         data=np.sum(weights * self.data[index], axis=axes_kernel) / np.sum(weights, axis=axes_kernel),
    #         grid={**self.grid, **grid},
    #     )

    @staticmethod
    @numba.njit(parallel=True, )
    def _calc_index_nearest_even_numba(
            grid: np.ndarray,
            grid_data: np.ndarray,
            mask_data: np.ndarray,
    ) -> np.ndarray:

        shape_grid_data = grid_data.shape
        shape_grid = grid.shape

        index_nearest_even = np.empty(shape_grid[:~0], dtype=numba.int64)

        for i_grid in numba.prange(shape_grid[~1]):
            distance_squared_min = np.inf
            for i_grid_data in numba.prange(shape_grid_data[~1]):
                if mask_data[i_grid_data]:
                    distance_squared = 0
                    for axis in numba.prange(shape_grid[~0]):
                        dist = grid[i_grid, axis] - grid_data[i_grid_data, axis]
                        distance_squared += dist * dist

                    if distance_squared < distance_squared_min:
                        index_nearest_even[i_grid] = i_grid_data
                        distance_squared_min = distance_squared

        return index_nearest_even

    def _calc_index_nearest_even(self, grid):

        grid_data = self.grid.subspace(grid)
        shape_grid_data = grid_data.shape
        # grid_data = {axis: np.broadcast_to(grid_data[axis], shape=shape_grid_data, subok=True) for axis in grid_data}
        # grid_data = {axis: grid_data[axis].combine_axes(axes=grid.keys(), axis_new='dummy') for axis in grid_data}
        grid_data = grid_data.flatten(axis_new='dummy')
        mask = np.indices(shape_grid_data.values()).sum(0) % 2 == 0
        mask = mask.reshape(-1)

        # grid = grid.broadcasted
        shape_grid = grid.shape
        grid = grid.flatten(axis_new='dummy')
        # grid = {axis: np.broadcast_to(grid[axis], shape=shape_grid, subok=True) for axis in grid}
        # grid = {axis: grid[axis].combine_axes(axes=grid.keys(), axis_new='dummy') for axis in grid}

        index = Array._calc_index_nearest_even_numba(
            grid=np.stack([d.value.data for d in grid.coordinates], axis=~0),
            grid_data=np.stack([d.value.data for d in grid_data.coordinates], axis=~0),
            mask_data=mask,
        )

        index = np.unravel_index(index, shape=tuple(shape_grid_data.values()))

        grid_nearest = type(grid)()
        for c, component in enumerate(grid_nearest):
            grid_nearest[component] = kgpy.labeled.Array(
                value=index[c].reshape(tuple(shape_grid.values())),
                axes=list(shape_grid.keys()),
            )

        return grid_nearest

        # return {axis: LabeledArray(i.reshape(tuple(shape_grid.values())), list(shape_grid.keys())) for axis, i in zip(shape_grid, index)}

    # def _calc_index_nearest_even_argmin(self, **grid: LabeledArray, ) -> typ.Dict[str, LabeledArray]:
    #
    #     grid_data = self.grid_broadcasted
    #
    #     shape_dummy = dict()
    #     for axis_name_data in grid_data:
    #         axis_names = grid_data[axis_name_data].axis_names
    #         for axis_name in grid:
    #
    #             if axis_name in axis_names:
    #                 axis_name_dummy = f'{axis_name}_dummy'
    #                 shape_dummy[axis_name_dummy] = self.shape[axis_name]
    #                 axis_index = axis_names.index(axis_name)
    #                 axis_names[axis_index] = axis_name_dummy
    #
    #     distance_squared = 0
    #     for axis_name in grid:
    #         distance_squared_axis = np.square(grid[axis_name] - grid_data[axis_name])
    #         distance_squared = distance_squared + distance_squared_axis
    #
    #     mask = LabeledArray(
    #         data=np.indices(shape_dummy.values()).sum(0) % 2 == 0,
    #         axis_names=list(shape_dummy.keys()),
    #     )
    #     mask = np.broadcast_to(mask, shape=distance_squared.shape, subok=True)
    #     distance_squared[~mask] = np.inf
    #
    #     distance_squared = distance_squared.combine_axes(axes=shape_dummy.keys(), axis_new='dummy')
    #     index = np.argmin(distance_squared, axis='dummy')
    #     index = np.unravel_index(index, shape_dummy)
    #
    #     index = {k[:~(len('_dummy') - 1)]: index[k] for k in index}
    #
    #     return index

    def interp_barycentric_linear(
            self: ArrayT,
            grid,
    ) -> ArrayT:

        grid_data = self.grid.subspace(grid)
        # todo: fix this!!!
        # for component in grid_data:
        #     for axis_other in grid_data[component].axis_names:
        #         if axis_other not in grid_data:
        #             raise ValueError('Attempting to broadcast interpolation along non-separable axis.')
        shape_grid_data = grid_data.shape
        grid_data = grid_data.broadcasted

        index_nearest = self._calc_index_nearest_even(grid)
        # index_nearest = self._calc_index_nearest_even_argmin(**grid)

        num_vertices = len(grid) + 1
        index = type(index_nearest)()
        axes_simplex = []
        for a, axis in enumerate(grid):
            axis_simplex = f'simplex_{axis}'
            axes_simplex.append(axis_simplex)
            simplex_axis = kgpy.labeled.Array.zeros(shape={axis_simplex: 2, 'vertices': num_vertices}, dtype=int)
            simplex_axis[dict(vertices=a+1)] = kgpy.labeled.Array(np.array([-1, 1], dtype=int), axes=[axis_simplex])
            index[axis] = index_nearest[axis] + simplex_axis

        shape_index = index.shape
        barycentric_transform_shape = shape_index.copy()
        barycentric_transform_shape['vertices'] = len(grid)
        barycentric_transform_shape['axis'] = len(grid)
        barycentric_transform = kgpy.labeled.Array.empty(barycentric_transform_shape)

        index_0 = {k: index[k][dict(vertices=0)] for k in index}
        index_1 = {k: index[k][dict(vertices=slice(1, None))] % shape_grid_data[k] for k in index}

        for a, axis in enumerate(grid):
            x0 = grid_data[axis][index_0]
            x1 = grid_data[axis][index_1]
            barycentric_transform[dict(axis=a)] = x1 - x0

        barycentric_transform = barycentric_transform.matrix_inverse(
            axis_rows='axis',
            axis_columns='vertices',
        )

        barycentric_coordinates = np.stack(
            arrays=[grid[axis] - grid_data[axis][index_nearest] for axis in grid],
            axis='axis',
        ).add_axes(axes_simplex + ['vertices'])

        barycentric_coordinates = barycentric_transform.matrix_multiply(
            barycentric_coordinates,
            axis_rows='axis',
            axis_columns='vertices',
        ).combine_axes(['vertices', 'axis'], axis_new='vertices')

        epsilon = 1e-15
        mask_inside = (-epsilon <= barycentric_coordinates) & (barycentric_coordinates <= 1 + epsilon)
        for axis in index:
            mask_inside = mask_inside & (index[axis][dict(vertices=slice(1, None))] >= 0)
            mask_inside = mask_inside & (index[axis][dict(vertices=slice(1, None))] < shape_grid_data[axis])
        mask_inside = np.all(mask_inside, axis='vertices')

        shape_weights = shape_index
        weights = kgpy.labeled.Array.empty(shape_weights)
        weights[dict(vertices=0)] = 1 - np.sum(barycentric_coordinates, axis='vertices')
        weights[dict(vertices=slice(1, None))] = barycentric_coordinates

        value = weights * self.value_broadcasted[{k: index[k] % shape_grid_data[k] for k in index}]
        value = np.nansum(value, axis='vertices')

        mask_inside = np.broadcast_to(mask_inside, shape=value.shape, subok=True)
        value[~mask_inside] = np.nan

        return Array(
            value=np.nanmean(value, axis=axes_simplex),
            grid=self.grid.from_dict({**self.grid.value, **grid.value}),
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

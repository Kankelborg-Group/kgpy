import typing as typ
import dataclasses
import numpy as np
import scipy.interpolate
import numba
import kgpy.labeled
import kgpy.grid

__all__ = [

]

GridT = typ.TypeVar('GridT', bound=kgpy.grid.Grid)
ArrayT = typ.TypeVar('ArrayT', bound='Array')


@dataclasses.dataclass
class Array(
    kgpy.mixin.Copyable,
    typ.Generic[GridT],
):

    value: kgpy.labeled.Array
    grid: GridT

    @property
    def shape(self: ArrayT) -> typ.Dict[str, int]:
        return self.grid.shape

    @property
    def value_broadcasted(self: ArrayT) -> ArrayT:
        return np.broadcast_to(self.value, shape=self.shape, subok=True)

    # @property
    # def axes_separable(self) -> typ.List[str]:
    #     axes = []
    #     grid = self.grid
    #     for axis in grid.value:
    #         if grid.value[axis].axis_names == [axis]:
    #             axes.append(axis)
    #     return axes

    @property
    def ndim(self: ArrayT) -> int:
        return self.grid.ndim

    @property
    def size(self: ArrayT) -> int:
        return self.grid.size

    def __eq__(self: ArrayT, other: ArrayT):
        if not super().__eq__(other):
            return False
        if not np.array_equal(self.value, other.value):
            return False
        if not self.grid == other.grid:
            return False
        return True

    def __getitem__(
            self: ArrayT,
            item: typ.Union[typ.Dict[str, typ.Union[int, slice, kgpy.labeled.AbstractArray]], kgpy.labeled.AbstractArray],
    ) -> ArrayT:
        grid = self.grid.broadcasted
        grid.value = {axis: grid.value[axis][item] for axis in grid.value}
        return type(self)(
            value=self.value[item],
            grid=grid,
        )

    # def calc_index_nearest(self, **grid: LabeledArray, ) -> typ.Dict[str, LabeledArray]:
    #
    #     grid_data = self.grid_normalized
    #
    #     shape_dummy = dict()
    #     for axis_name_data in grid_data:
    #         axis_names = grid_data[axis_name_data].axis_names
    #         for axis_name in grid:
    #             if axis_name in axis_names:
    #                 axis_name_dummy = f'{axis_name}_dummy'
    #                 shape_dummy[axis_name_dummy] = self.shape[axis_name]
    #                 axis_index = axis_names.index(axis_name)
    #                 axis_names[axis_index] = axis_name_dummy
    #
    #     distance_squared = 0
    #     for axis_name in grid:
    #         distance_squared = distance_squared + np.square(grid[axis_name] - grid_data[axis_name])
    #
    #     distance_squared = distance_squared.combine_axes(axes=shape_dummy.keys(), axis_new='dummy')
    #     index = np.argmin(distance_squared, axis='dummy')
    #     index = np.unravel_index(index, shape_dummy)
    #
    #     index = {k[:~(len('_dummy') - 1)]: index[k] for k in index}
    #
    #     return index
    #
    # def interp_nearest(self, **grid: LabeledArray) -> 'DataArray':
    #     return DataArray(
    #         data=self.data[self.calc_index_nearest(**grid)],
    #         grid={**self.grid, **grid},
    #     )
    #
    # def calc_index_lower(self, **grid: LabeledArray, ) -> typ.Dict[str, LabeledArray]:
    #
    #     grid_data = self.grid_normalized
    #
    #     shape_dummy = dict()
    #     for axis_name_data in grid_data:
    #         axis_names = grid_data[axis_name_data].axis_names
    #         for axis_name in grid:
    #             if axis_name in axis_names:
    #                 axis_name_dummy = f'{axis_name}_dummy'
    #                 shape_dummy[axis_name_dummy] = self.shape[axis_name]
    #                 axis_index = axis_names.index(axis_name)
    #                 axis_names[axis_index] = axis_name_dummy
    #
    #     distance_squared = 0
    #     for axis_name in grid:
    #         d = grid_data[axis_name] - grid[axis_name]
    #         d[d > 0] = -np.inf
    #         distance_squared = distance_squared + d
    #
    #     distance_squared = distance_squared.combine_axes(axes=shape_dummy.keys(), axis_new='dummy')
    #     index = np.argmax(distance_squared, axis='dummy')
    #     index = np.unravel_index(index, shape_dummy)
    #
    #     for axis in index:
    #         index_max = shape_dummy[axis] - 2
    #         index[axis][index[axis] > index_max] = index_max
    #
    #     index = {k[:~(len('_dummy') - 1)]: index[k] for k in index}
    #
    #     return index
    #
    # def _interp_linear_1d_recursive(
    #         self,
    #         grid: typ.Dict[str, LabeledArray],
    #         index_lower: typ.Dict[str, LabeledArray],
    #         axis_stack: typ.List[str],
    # ) -> LabeledArray:
    #
    #     axis = axis_stack.pop(0)
    #
    #     index_upper = copy.deepcopy(index_lower)
    #     index_upper[axis] = index_upper[axis] + 1
    #
    #     x = grid[axis]
    #
    #     grid_data = self.grid_broadcasted
    #     x0 = grid_data[axis][index_lower]
    #     x1 = grid_data[axis][index_upper]
    #
    #     if axis_stack:
    #         y0 = self._interp_linear_1d_recursive(grid=grid, index_lower=index_lower, axis_stack=axis_stack.copy())
    #         y1 = self._interp_linear_1d_recursive(grid=grid, index_lower=index_upper, axis_stack=axis_stack.copy())
    #
    #     else:
    #         data = self.data_broadcasted
    #         y0 = data[index_lower]
    #         y1 = data[index_upper]
    #
    #     result = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    #
    #     return result
    #
    # def interp_linear(self, **grid: LabeledArray, ) -> 'DataArray':
    #
    #     return DataArray(
    #         data=self._interp_linear_1d_recursive(
    #             grid=grid,
    #             index_lower=self.calc_index_lower(**grid),
    #             axis_stack=list(grid.keys()),
    #         ),
    #         grid={**self.grid, **grid}
    #     )
    #
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

    def _calc_index_nearest_even(self, grid: GridT) -> GridT:

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
            grid: GridT,
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

    def interp_barycentric_linear_scipy(self: ArrayT, grid: GridT):

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
            grid: GridT,
    ) -> ArrayT:
        return self.interp_barycentric_linear(grid=grid)

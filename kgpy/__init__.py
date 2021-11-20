"""
kgpy root package
"""
import typing as typ
import dataclasses
import copy
import numpy as np
import numpy.typing
import numba

__all__ = [
    'linspace', 'midspace',
    'Name',
    'fft',
    'rebin',
]

import scipy.interpolate


@dataclasses.dataclass
class Name:
    """
    Representation of a hierarchical namespace.
    Names are a composition of a parent, which is also a name, and a base which is a simple string.
    The string representation of a name is <parent>.base, where <parent> is the parent's string expansion.
    """

    base: str = ''  #: Base of the name, this string will appear last in the string representation
    parent: 'typ.Optional[Name]' = None     #: Parent string of the name, this itself also a name

    def copy(self):
        if self.parent is not None:
            parent = self.parent.copy()
        else:
            parent = self.parent
        return type(self)(
            base=self.base,
            parent=parent,
        )

    def __add__(self, other: str) -> 'Name':
        """
        Quickly create the name of a child's name by adding a string to the current instance.
        Adding a string to a name instance returns
        :param other: A string representing the basename of the new Name instance.
        :return: A new `kgpy.Name` instance with the `self` as the `parent` and `other` as the `base`.
        """
        return type(self)(base=other, parent=self)

    def __repr__(self):
        if self.parent is not None:
            return self.parent.__repr__() + '.' + self.base

        else:
            return self.base


import kgpy.mixin


def rebin(arr: np.ndarray, scale_dims: typ.Tuple[int, ...]) -> np.ndarray:
    """
    Increases the size of an array by scale_dims in each i dimension by repeating each value scale_dims[i] times along
    that axis.

    :param arr: Array to modify
    :param scale_dims: Tuple with length ``arr.ndim`` specifying the size increase in each axis.
    :return: The resized array
    """
    new_arr = np.broadcast_to(arr, scale_dims + arr.shape)
    start_axes = np.arange(arr.ndim)
    new_axes = 2 * start_axes + 1
    new_arr = np.moveaxis(new_arr, start_axes, new_axes)

    new_shape = np.array(arr.shape) * np.array(scale_dims)
    new_arr = np.reshape(new_arr, new_shape)
    return new_arr


def linspace(start: np.ndarray, stop: np.ndarray, num: int, axis: int = 0) -> numpy.typing.ArrayLike:
    """
    A modified version of :func:`numpy.linspace()` that returns a value in the center of the range between `start`
    and `stop` if `num == 1` unlike :func:`numpy.linspace` which would just return `start`.
    This function is often helfpul when creating a grid.
    Sometimes you want to test with only a single element, but you want that element to be in the center of the range
    and not off to one side.

    :param start: The starting value of the sequence.
    :param stop: The end value of the sequence, must be broadcastable with `start`.
    :param num: Number of samples to generate for this sequence.
    :param axis: The axis in the result used to store the samples. The default is the first axis.
    :return: An array the size of the broadcasted shape of `start` and `stop` with an additional dimension of length
        `num`.
    """
    if num == 1:
        return np.expand_dims((start + stop) / 2, axis=axis)
    else:
        return np.linspace(start=start, stop=stop, num=num, axis=axis)


def midspace(start: np.ndarray, stop: np.ndarray, num: int, axis: int = 0) -> numpy.typing.ArrayLike:
    """
    A modified version of :func:`numpy.linspace` that selects cell centers instead of cell edges.

    :param start:
    :param stop:
    :param num:
    :param axis:
    :return:
    """
    a = np.linspace(start=start, stop=stop, num=num + 1, axis=axis)
    i0 = [slice(None)] * a.ndim
    i1 = i0.copy()
    i0[axis] = slice(None, ~0)
    i1[axis] = slice(1, None)
    return (a[i0] + a[i1]) / 2


def rms(a: np.ndarray, axis: typ.Optional[typ.Union[int, typ.Sequence[int]]] = None):
    return np.sqrt(np.mean(np.square(a), axis=axis))


def take(
        a: numpy.typing.ArrayLike,
        key: typ.Union[numpy.typing.ArrayLike, slice],
        axis: int = 0,
) -> numpy.typing.ArrayLike:
    if isinstance(key, slice):
        return a[(slice(None),) * (axis % a.ndim) + (key,)]
    else:
        return np.take(a=a, indices=key, axis=axis)


def takes(
        a: numpy.typing.ArrayLike,
        keys: typ.Sequence[typ.Union[numpy.typing.ArrayLike, slice]],
        axes: typ.Sequence[int],
) -> numpy.typing.ArrayLike:
    for key, axis in zip(keys, axes):
        a = take(a=a, key=key, axis=axis)
    return a







@dataclasses.dataclass
class DataArray(kgpy.mixin.Copyable):

    data: LabeledArray
    grid: typ.Dict[str, LabeledArray]

    @property
    def grid_normalized(self) -> typ.Dict[str, LabeledArray]:
        # grid_normalized = self.grid.copy()
        grid_normalized = copy.deepcopy(self.grid)
        for axis_name in self.data.axis_names:
            if axis_name not in grid_normalized:
                grid_normalized[axis_name] = LabeledArray.arange(stop=self.data.shape[axis_name], axis=axis_name)
        return grid_normalized

    @property
    def shape(self) -> typ.Dict[str, int]:
        return LabeledArray.broadcast_shapes(*self.grid_normalized.values())

    @property
    def data_broadcasted(self):
        return np.broadcast_to(self.data, shape=self.shape, subok=True)

    @property
    def grid_broadcasted(self) -> typ.Dict[str, LabeledArray]:
        grid = self.grid_normalized
        shape = self.shape
        return {k: np.broadcast_to(grid[k], shape=shape, subok=True) for k in grid}

    @property
    def axes_separable(self) -> typ.List[str]:
        axes = []
        grid = self.grid_normalized
        for axis in grid:
            if grid[axis].axis_names == [axis]:
                axes.append(axis)
        return axes

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.array(self.shape.values()).prod()

    def __eq__(self, other: 'DataArray'):
        if not super().__eq__(other):
            return False
        if not np.array_equal(self.data, other.data):
            return False
        for g in self.grid:
            if not np.array_equal(self.grid[g], other.grid[g]):
                return False
        return True

    def __getitem__(
            self,
            item: typ.Union[typ.Dict[str, typ.Union[int, slice, LabeledArray]], LabeledArray],
    ) -> 'DataArray':
        grid = self.grid_broadcasted
        return DataArray(
            data=self.data[item],
            grid={k: grid[k][item] for k in grid},
        )

    def calc_index_nearest(self, **grid: LabeledArray, ) -> typ.Dict[str, LabeledArray]:

        grid_data = self.grid_normalized

        shape_dummy = dict()
        for axis_name_data in grid_data:
            axis_names = grid_data[axis_name_data].axis_names
            for axis_name in grid:
                if axis_name in axis_names:
                    axis_name_dummy = f'{axis_name}_dummy'
                    shape_dummy[axis_name_dummy] = self.shape[axis_name]
                    axis_index = axis_names.index(axis_name)
                    axis_names[axis_index] = axis_name_dummy

        distance_squared = 0
        for axis_name in grid:
            distance_squared = distance_squared + np.square(grid[axis_name] - grid_data[axis_name])

        distance_squared = distance_squared.combine_axes(axes=shape_dummy.keys(), axis_new='dummy')
        index = np.argmin(distance_squared, axis='dummy')
        index = np.unravel_index(index, shape_dummy)

        index = {k[:~(len('_dummy') - 1)]: index[k] for k in index}

        return index

    def interp_nearest(self, **grid: LabeledArray) -> 'DataArray':
        return DataArray(
            data=self.data[self.calc_index_nearest(**grid)],
            grid={**self.grid, **grid},
        )

    def calc_index_lower(self, **grid: LabeledArray, ) -> typ.Dict[str, LabeledArray]:

        grid_data = self.grid_normalized

        shape_dummy = dict()
        for axis_name_data in grid_data:
            axis_names = grid_data[axis_name_data].axis_names
            for axis_name in grid:
                if axis_name in axis_names:
                    axis_name_dummy = f'{axis_name}_dummy'
                    shape_dummy[axis_name_dummy] = self.shape[axis_name]
                    axis_index = axis_names.index(axis_name)
                    axis_names[axis_index] = axis_name_dummy

        distance_squared = 0
        for axis_name in grid:
            d = grid_data[axis_name] - grid[axis_name]
            d[d > 0] = -np.inf
            distance_squared = distance_squared + d

        distance_squared = distance_squared.combine_axes(axes=shape_dummy.keys(), axis_new='dummy')
        index = np.argmax(distance_squared, axis='dummy')
        index = np.unravel_index(index, shape_dummy)

        for axis in index:
            index_max = shape_dummy[axis] - 2
            index[axis][index[axis] > index_max] = index_max

        index = {k[:~(len('_dummy') - 1)]: index[k] for k in index}

        return index

    def _interp_linear_1d_recursive(
            self,
            grid: typ.Dict[str, LabeledArray],
            index_lower: typ.Dict[str, LabeledArray],
            axis_stack: typ.List[str],
    ) -> LabeledArray:

        axis = axis_stack.pop(0)

        index_upper = copy.deepcopy(index_lower)
        index_upper[axis] = index_upper[axis] + 1

        x = grid[axis]

        grid_data = self.grid_broadcasted
        x0 = grid_data[axis][index_lower]
        x1 = grid_data[axis][index_upper]

        if axis_stack:
            y0 = self._interp_linear_1d_recursive(grid=grid, index_lower=index_lower, axis_stack=axis_stack.copy())
            y1 = self._interp_linear_1d_recursive(grid=grid, index_lower=index_upper, axis_stack=axis_stack.copy())

        else:
            data = self.data_broadcasted
            y0 = data[index_lower]
            y1 = data[index_upper]

        result = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

        return result

    def interp_linear(self, **grid: LabeledArray, ) -> 'DataArray':

        return DataArray(
            data=self._interp_linear_1d_recursive(
                grid=grid,
                index_lower=self.calc_index_lower(**grid),
                axis_stack=list(grid.keys()),
            ),
            grid={**self.grid, **grid}
        )

    def interp_idw(
            self,
            grid: typ.Dict[str, LabeledArray],
            power: float = 2,
    ) -> 'DataArray':

        grid_data = self.grid_broadcasted

        index = self.calc_index_lower(**grid)

        axes_kernel = []
        for axis in grid:
            axis_kernel = f'kernel_{axis}'
            axes_kernel.append(axis_kernel)
            index[axis] = index[axis] + LabeledArray.arange(stop=2, axis=axis_kernel)

        distance_to_power = 0
        for axis in grid:
            distance_to_power = distance_to_power + np.abs(grid_data[axis][index] - grid[axis]) ** power
        distance = distance_to_power ** (1 / power)

        weights = 1 / distance

        return DataArray(
            data=np.sum(weights * self.data[index], axis=axes_kernel) / np.sum(weights, axis=axes_kernel),
            grid={**self.grid, **grid},
        )

    @staticmethod
    @numba.njit(parallel=True, )
    def _calc_index_nearest_even_numba(
            grid: np.ndarray,
            grid_data: np.ndarray,
            mask_data: np.ndarray,
    ):

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

    def _calc_index_nearest_even(self, **grid: LabeledArray, ) -> typ.Dict[str, LabeledArray]:

        grid_data = {axis: self.grid[axis] for axis in grid}
        shape_grid_data = LabeledArray.shape_broadcasted(*grid_data.values())
        grid_data = {axis: np.broadcast_to(grid_data[axis], shape=shape_grid_data, subok=True) for axis in grid_data}
        grid_data = {axis: grid_data[axis].combine_axes(axes=grid.keys(), axis_new='dummy') for axis in grid_data}
        mask = np.indices(shape_grid_data.values()).sum(0) % 2 == 0
        mask = mask.reshape(-1)

        shape_grid = LabeledArray.shape_broadcasted(*grid.values())
        grid = {axis: np.broadcast_to(grid[axis], shape=shape_grid, subok=True) for axis in grid}
        grid = {axis: grid[axis].combine_axes(axes=grid.keys(), axis_new='dummy') for axis in grid}

        index = DataArray._calc_index_nearest_even_numba(
            grid=np.stack([d.data for d in grid.values()], axis=~0),
            grid_data=np.stack([d.data for d in grid_data.values()], axis=~0),
            mask_data=mask,
        )

        index = np.unravel_index(index, shape=tuple(shape_grid_data.values()))

        return {axis: LabeledArray(i.reshape(tuple(shape_grid.values())), list(shape_grid.keys())) for axis, i in zip(shape_grid, index)}

    def _calc_index_nearest_even_argmin(self, **grid: LabeledArray, ) -> typ.Dict[str, LabeledArray]:

        grid_data = self.grid_broadcasted

        shape_dummy = dict()
        for axis_name_data in grid_data:
            axis_names = grid_data[axis_name_data].axis_names
            for axis_name in grid:
                if axis_name in axis_names:
                    axis_name_dummy = f'{axis_name}_dummy'
                    shape_dummy[axis_name_dummy] = self.shape[axis_name]
                    axis_index = axis_names.index(axis_name)
                    axis_names[axis_index] = axis_name_dummy

        distance_squared = 0
        for axis_name in grid:
            distance_squared_axis = np.square(grid[axis_name] - grid_data[axis_name])
            distance_squared = distance_squared + distance_squared_axis

        mask = LabeledArray(
            data=np.indices(shape_dummy.values()).sum(0) % 2 == 0,
            axis_names=list(shape_dummy.keys()),
        )
        mask = np.broadcast_to(mask, shape=distance_squared.shape, subok=True)
        distance_squared[~mask] = np.inf

        distance_squared = distance_squared.combine_axes(axes=shape_dummy.keys(), axis_new='dummy')
        index = np.argmin(distance_squared, axis='dummy')
        index = np.unravel_index(index, shape_dummy)

        index = {k[:~(len('_dummy') - 1)]: index[k] for k in index}

        return index

    def interp_barycentric_linear(
            self,
            grid: typ.Dict[str, LabeledArray],
    ) -> 'DataArray':

        grid_data = {axis: self.grid[axis] for axis in grid}
        for axis in grid_data:
            for axis_other in grid_data[axis].axis_names:
                if axis_other not in grid_data:
                    raise ValueError('Attempting to broadcast interpolation along non-separable axis.')
        shape_grid_data = LabeledArray.shape_broadcasted(*grid_data.values())
        grid_data = {axis: np.broadcast_to(grid_data[axis], shape=shape_grid_data, subok=True) for axis in grid_data}

        index_nearest = self._calc_index_nearest_even(**grid)
        # index_nearest = self._calc_index_nearest_even_argmin(**grid)

        num_vertices = len(grid) + 1
        index = index_nearest.copy()
        axes_simplex = []
        for a, axis in enumerate(grid):
            axis_simplex = f'simplex_{axis}'
            axes_simplex.append(axis_simplex)
            simplex_axis = LabeledArray.zeros(shape={axis_simplex: 2, 'vertices': num_vertices}, dtype=int)
            simplex_axis[dict(vertices=a+1)] = LabeledArray(np.array([-1, 1], dtype=int), axis_names=[axis_simplex])
            index[axis] = (index_nearest[axis] + simplex_axis)

        shape_index = LabeledArray.broadcast_shapes(*index.values())
        barycentric_transform_shape = shape_index.copy()
        barycentric_transform_shape['vertices'] = len(grid)
        barycentric_transform_shape['axis'] = len(grid)
        barycentric_transform = LabeledArray.empty(barycentric_transform_shape)

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
        weights = LabeledArray.empty(shape_weights)
        weights[dict(vertices=0)] = 1 - np.sum(barycentric_coordinates, axis='vertices')
        weights[dict(vertices=slice(1, None))] = barycentric_coordinates

        data = weights * self.data_broadcasted[{k: index[k] % shape_grid_data[k] for k in index}]
        data = np.nansum(data, axis='vertices')

        mask_inside = np.broadcast_to(mask_inside, shape=data.shape, subok=True)
        data[~mask_inside] = np.nan

        return DataArray(
            data=np.nanmean(data, axis=axes_simplex),
            grid={**self.grid, **grid},
        )

    def interp_barycentric_linear_scipy(self, grid: typ.Dict[str, LabeledArray]):

        axes_uninterpolated = self.grid_normalized.keys() - grid.keys()
        print('axes_uninterpolated', axes_uninterpolated)

        shape_grid = LabeledArray.broadcast_shapes(*grid.values())
        grid = {k: np.broadcast_to(grid[k], shape=shape_grid, subok=True) for k in grid}

        data_interp = scipy.interpolate.griddata(
            points=tuple(val.data.reshape(-1) for val in self.grid_broadcasted.values()),
            values=self.data_broadcasted.data.reshape(-1),
            xi=tuple(val.data.reshape(-1) for val in grid.values()),
        )

        data_interp = LabeledArray(
            data=data_interp.reshape(tuple(shape_grid.values())),
            axis_names=list(grid.keys()),
        )

        return DataArray(
            data=data_interp,
            grid=grid,
        )

    def __call__(
            self,
            **grid: LabeledArray,
    ) -> 'DataArray':
        return self.interp_barycentric_linear(grid=grid)

    def view(self) -> 'DataArray':
        other = super().view()  # type: DataArray
        other.data = self.data
        other.grid = self.grid
        return other

    def copy(self) -> 'DataArray':
        other = super().copy()      # type: DataArray
        other.data = self.data.copy()
        other.grid = copy.deepcopy(self.grid)
        return other

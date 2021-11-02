"""
kgpy root package
"""
import typing as typ
import dataclasses
import copy
import numpy as np
import numpy.typing

__all__ = [
    'linspace', 'midspace',
    'Name',
    'fft',
    'rebin',
]


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


@dataclasses.dataclass(eq=False)
class LabeledArray(
    kgpy.mixin.Copyable,
    np.lib.mixins.NDArrayOperatorsMixin,
):
    data: numpy.typing.ArrayLike
    axis_names: typ.List[str]

    def __post_init__(self):
        if np.ndim(self.data) != len(self.axis_names):
            raise ValueError('The number of axis names must match the number of dimensions.')
        if len(self.axis_names) != len(set(self.axis_names)):
            raise ValueError('Each axis name must be unique.')

    @classmethod
    def empty(cls, shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> 'LabeledArray':
        return LabeledArray(
            data=np.empty(shape=tuple(shape.values()), dtype=dtype),
            axis_names=list(shape.keys()),
        )

    @classmethod
    def zeros(cls, shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> 'LabeledArray':
        return LabeledArray(
            data=np.zeros(shape=tuple(shape.values()), dtype=dtype),
            axis_names=list(shape.keys()),
        )

    @classmethod
    def ones(cls, shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> 'LabeledArray':
        return LabeledArray(
            data=np.ones(shape=tuple(shape.values()), dtype=dtype),
            axis_names=list(shape.keys()),
        )

    @property
    def shape(self) -> typ.Dict[str, int]:
        shape = dict()
        for i in range(np.ndim(self.data)):
            shape[self.axis_names[i]] = self.data.shape[i]
        return shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def broadcast_shapes(cls, *arrs: 'LabeledArray') -> typ.Dict[str, int]:
        shape = dict()
        for a in arrs:
            if hasattr(a, 'shape'):
                a_shape = a.shape
                for k in a_shape:
                    if k in shape:
                        shape[k] = max(shape[k], a_shape[k])
                    else:
                        shape[k] = a_shape[k]
        return shape

    def shape_broadcasted(self, *arrs: 'LabeledArray'):
        return self.broadcast_shapes(self, *arrs)

    def _data_aligned(self, shape: typ.Dict[str, int]) -> numpy.typing.ArrayLike:
        ndim_missing = len(shape) - np.ndim(self.data)
        data = np.expand_dims(self.data, tuple(~np.arange(ndim_missing)))
        source = []
        destination = []
        for axis_index, axis_name in enumerate(self.axis_names):
            source.append(axis_index)
            destination.append(list(shape.keys()).index(axis_name))
        data = np.moveaxis(a=data, source=source, destination=destination)
        return data

    @classmethod
    def arange(
            cls,
            axis: str,
            start: int = 0,
            *args,
            **kwargs,
    ) -> 'LabeledArray':
        return LabeledArray(
            data=np.arange(start=start, *args, **kwargs),
            axis_names=[axis],
        )

    @classmethod
    def linspace(
            cls,
            start: typ.Union[float, 'LabeledArray'],
            stop: typ.Union[float, 'LabeledArray'],
            num: int,
            axis: str,
            endpoint: bool = True,
            dtype: numpy.typing.DTypeLike = None,

    ) -> 'LabeledArray':
        if not isinstance(start, LabeledArray):
            if np.isscalar(start):
                start = np.array(start)
            start = LabeledArray(data=start, axis_names=[])
        if not isinstance(stop, LabeledArray):
            stop = LabeledArray(data=stop, axis_names=[])
        shape = start.shape_broadcasted(stop)

        if axis in shape:
            raise ValueError('Axis already defined, pick a new axis.')

        shape[axis] = num

        return LabeledArray(
            data=np.linspace(
                start=start._data_aligned(shape)[..., 0],
                stop=stop._data_aligned(shape)[..., 0],
                num=num,
                endpoint=endpoint,
                dtype=dtype,
                axis=~0,
            ),
            axis_names=list(shape.keys()),
        )

    def add_axes(self, axes: typ.List) -> 'LabeledArray':
        shape_new = {axis: 1 for axis in axes}
        shape = {**self.shape, **shape_new}
        return LabeledArray(
            data=self._data_aligned(shape),
            axis_names=list(shape.keys()),
        )

    def combine_axes(
            self,
            axes: typ.Sequence[str],
            axis_new: typ.Optional[str] = None,
    ) -> 'LabeledArray':
        axes = list(axes)
        if axis_new is None:
            axis_new = ''.join(axes)

        axes_preserved = list(np.setdiff1d(self.axis_names, axes))
        axes_all = axes_preserved + axes

        source = []
        destination = []
        for axis in axes_all:
            source.append(self.axis_names.index(axis))
            destination.append(axes_all.index(axis))

        axes_new = axes_preserved + [axis_new]
        shape_new = dict()
        for axis in axes_preserved:
            shape_new[axis] = self.shape[axis]
        shape_new[axis_new] = -1

        return LabeledArray(
            data=np.moveaxis(self.data, source=source, destination=destination).reshape(tuple(shape_new.values())),
            axis_names=axes_new,
        )

    def matrix_multiply(self, other: 'LabeledArray', axis_rows: str, axis_columns: str) -> 'LabeledArray':
        shape = LabeledArray.shape_broadcasted(other)
        shape_rows = shape.pop(axis_rows)
        shape_columns = shape.pop(axis_columns)
        shape = {**shape, axis_rows: shape_rows, axis_columns: shape_columns}

        data_self = self._data_aligned(shape)
        data_other = other._data_aligned(shape)

        return LabeledArray(
            data=np.matmul(data_self, data_other),
            axis_names=list(shape.keys()),
        )

    def matrix_inverse(self, axis_rows: str, axis_columns: str) -> 'LabeledArray':
        data = np.moveaxis(
            a=self.data,
            source=[self.axis_names.index(axis_rows), self.axis_names.index(axis_columns)],
            destination=[~1, ~0],
        )

        axis_names_new = self.axis_names.copy()
        axis_names_new.remove(axis_rows)
        axis_names_new.remove(axis_columns)
        axis_names_new += [axis_rows, axis_columns]

        return LabeledArray(
            data=np.linalg.inv(data),
            axis_names=axis_names_new,
        )

    def __array_ufunc__(
            self,
            function: np.ufunc,
            method: str,
            *inputs: 'LabeledArray',
            **kwargs: typ.Any,
    ):
        shape = self.broadcast_shapes(*inputs)
        inputs = [LabeledArray(data=inp, axis_names=[]) if np.isscalar(inp) else inp for inp in inputs ]
        inputs = [inp._data_aligned(shape) for inp in inputs]

        for inp in inputs:
            if hasattr(inp, '__array_ufunc__'):
                result = inp.__array_ufunc__(function, method, *inputs, **kwargs)
                if result is not NotImplemented:
                    return LabeledArray(
                        data=result,
                        axis_names=list(shape.keys()),
                    )
        raise ValueError

    def __array_function__(
            self,
            function: typ.Callable,
            types: typ.Collection,
            args: typ.Tuple,
            kwargs: typ.Dict[str, typ.Any],
    ):
        if function is np.broadcast_to:
            args = list(args)
            if 'subok' in kwargs:
                subok = kwargs['subok']
            else:
                subok = args.pop()
            if 'shape' in kwargs:
                shape = kwargs['shape']
            else:
                shape = args.pop()
            if 'array' in kwargs:
                array = kwargs['array']
            else:
                array = args.pop()
            return LabeledArray(
                data=np.broadcast_to(array=array._data_aligned(shape), shape=tuple(shape.values()), subok=subok),
                axis_names=list(shape.keys())
            )
        elif function is np.result_type:
            return type(self)
        elif function is np.unravel_index:
            args = list(args)
            if 'shape' in kwargs:
                shape = kwargs['shape']
            else:
                shape = args.pop()

            if 'indices' in kwargs:
                indices = kwargs['indices'].data
            else:
                indices = args.pop().data

            result_data = np.unravel_index(indices=indices, shape=tuple(shape.values()))
            result = dict()
            for axis, data in zip(shape, result_data):
                result[axis] = LabeledArray(
                    data=data,
                    axis_names=self.axis_names,
                )
            return result

        elif function is np.linalg.inv:
            raise ValueError(f'{function} is unsupported, use kgpy.LabelArray.matrix_inverse() instead.')

        elif function is np.stack:
            if 'arrays' in kwargs:
                arrays = kwargs.pop('arrays')
            else:
                arrays = args[0]    # type: typ.List[LabeledArray]

            if 'axis' in kwargs:
                axis = kwargs.pop('axis')
            else:
                raise ValueError('axis must be specified')

            shape = self.broadcast_shapes(*arrays)
            arrays = [arr._data_aligned(shape) for arr in arrays]

            return LabeledArray(
                data=np.stack(arrays=arrays, axis=0, **kwargs),
                axis_names=[axis] + self.axis_names,
            )

        elif function in [
            np.ndim,
            np.argmin,
            np.nanargmin,
            np.min,
            np.nanmin,
            np.argmax,
            np.nanargmax,
            np.max,
            np.nanmax,
            np.sum,
            np.nansum,
            np.mean,
            np.nanmean,
            np.median,
            np.nanmedian,
            np.percentile,
            np.nanpercentile,
            np.all,
            np.any,
            np.array_equal,
            np.isclose,
        ]:

            labeled_arrays = [arg for arg in args if isinstance(arg, LabeledArray)]
            labeled_arrays += [kwargs[k] for k in kwargs if isinstance(kwargs[k], LabeledArray)]
            shape = LabeledArray.broadcast_shapes(*labeled_arrays)
            axis_names = list(shape.keys())

            args = tuple(arg._data_aligned(shape) if isinstance(arg, LabeledArray) else arg for arg in args)
            kwargs = {k: kwargs[k]._data_aligned(shape) if isinstance(kwargs[k], LabeledArray) else kwargs[k] for k in kwargs}
            types = tuple(type(arg) for arg in args if getattr(arg, '__array_function__', None) is not None)

            result_axis_names = axis_names.copy()
            if function is not np.isclose:
                if 'keepdims' not in kwargs:
                    if 'axis' in kwargs:
                        if np.isscalar(kwargs['axis']):
                            result_axis_names.remove(kwargs['axis'])
                        else:
                            for axis in kwargs['axis']:
                                result_axis_names.remove(axis)
                    else:
                        result_axis_names = []

            if 'axis' in kwargs:
                if np.isscalar(kwargs['axis']):
                    kwargs['axis'] = axis_names.index(kwargs['axis'])
                else:
                    kwargs['axis'] = tuple(axis_names.index(ax) for ax in kwargs['axis'])

            data = self.data.__array_function__(function, types, args, kwargs)

            if function in [
                np.all,
                np.any,
                np.array_equal,
            ]:
                return data
            else:
                return type(self)(
                    data=data,
                    axis_names=result_axis_names,
                )
        else:
            raise ValueError('Unsupported function')

    def __getitem__(
            self,
            item: typ.Union[typ.Dict[str, typ.Union[int, slice, 'LabeledArray']], 'LabeledArray'],
    ) -> 'LabeledArray':

        if isinstance(item, LabeledArray):
            shape = self.shape_broadcasted(item)
            return LabeledArray(
                data=self._data_aligned(shape)[item._data_aligned(shape)],
                axis_names=['boolean', ],
            )

        else:
            axis_names_advanced = []
            axis_indices_advanced = []
            item_advanced = dict()
            for axis in item:
                if isinstance(item[axis], LabeledArray):
                    axis_names_advanced.append(axis)
                    axis_indices_advanced.append(self.axis_names.index(axis))
                    item_advanced[axis] = item[axis]

            shape_advanced = self.broadcast_shapes(*item_advanced.values())

            data = np.moveaxis(
                a=self.data,
                source=axis_indices_advanced,
                destination=list(range(len(axis_indices_advanced))),
            )

            axis_names = list(self.axis_names)
            for a, axis in enumerate(axis_names_advanced):
                axis_names.remove(axis)
                axis_names.insert(a, axis)

            axis_names_new = axis_names.copy()
            index = [slice(None)] * self.ndim
            for axis_name in item:
                item_axis = item[axis_name]
                if isinstance(item_axis, LabeledArray):
                    item_axis = item_axis._data_aligned(shape_advanced)
                index[axis_names.index(axis_name)] = item_axis
                if not isinstance(item_axis, slice):
                    axis_names_new.remove(axis_name)

            return LabeledArray(
                data=data[tuple(index)],
                axis_names=list(shape_advanced.keys()) + axis_names_new,
            )

    def __setitem__(
            self,
            key: typ.Union[typ.Dict[str, typ.Union[int, slice, 'LabeledArray']], 'LabeledArray'],
            value: typ.Union[float, 'LabeledArray'],
    ) -> typ.NoReturn:

        if isinstance(key, LabeledArray):
            shape = self.shape_broadcasted(key)
            self._data_aligned(shape)[key._data_aligned(shape)] = value

        else:
            index = [slice(None)] * self.ndim
            for axis_name in key:
                item_axis = key[axis_name]
                if isinstance(item_axis, LabeledArray):
                    item_axis = item_axis._data_aligned(self.shape_broadcasted(item_axis))
                index[self.axis_names.index(axis_name)] = item_axis

            self.data[tuple(index)] = value

    def view(self) -> 'LabeledArray':
        other = super().view()      # type: LabeledArray
        other.data = self.data
        other.axis_names = self.axis_names
        return other

    def copy(self) -> 'LabeledArray':
        other = super().copy()      # type: LabeledArray
        other.data = self.data.copy()
        other.axis_names = copy.deepcopy(self.axis_names)
        return other


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
        for axis_name in grid:
            if axis_name in grid_data:
                axis_names = grid_data[axis_name].axis_names
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

    def interp_rbf_gaussian(
            self,
            grid: typ.Dict[str, LabeledArray],
            width: typ.Union[float, typ.Dict[str, float]] = 1,
    ) -> 'DataArray':

        grid_data = self.grid_broadcasted

        index = self.calc_index_lower(**grid)

        axes_kernel = []
        for axis in grid:
            axis_kernel = f'kernel_{axis}'
            axes_kernel.append(axis_kernel)
            index[axis] = index[axis] + LabeledArray.arange(stop=2, axis=axis_kernel)

        print('index', index)

        power = 2
        distance_to_power = 0
        for axis in grid:
            distance_to_power = distance_to_power + np.abs(grid_data[axis][index] - grid[axis]) ** power
        distance = distance_to_power ** (1 / power)

        print('distance', distance.shape)

        weights = np.exp(-np.square(distance / width) / 2) / (np.sqrt(2 * np.pi) * width)

        return DataArray(
            data=np.sum(weights * self.data[index], axis=axes_kernel) / np.sum(weights, axis=axes_kernel),
            # data=np.sum(weights * self.data[index], axis=axes_kernel),
            grid={**self.grid, **grid},
        )


    def __call__(
            self,
            mode: str = 'linear',
            **grid: typ.Optional[numpy.typing.ArrayLike],
    ):
        pass

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

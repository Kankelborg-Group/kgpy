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


@dataclasses.dataclass
class LabeledArray(
    kgpy.mixin.Copyable,
    np.lib.mixins.NDArrayOperatorsMixin,
):
    data: numpy.typing.ArrayLike
    axis_names: typ.Tuple[str]

    def __post_init__(self):
        if np.ndim(self.data) != len(self.axis_names):
            raise ValueError('The number of axis names must match the number of dimensions.')

    @classmethod
    def empty(cls, shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> 'LabeledArray':
        return LabeledArray(
            data=np.empty(shape=tuple(shape.values()), dtype=dtype),
            axis_names=tuple(shape.keys()),
        )

    @classmethod
    def zeros(cls, shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> 'LabeledArray':
        return LabeledArray(
            data=np.zeros(shape=tuple(shape.values()), dtype=dtype),
            axis_names=tuple(shape.keys()),
        )

    @classmethod
    def ones(cls, shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> 'LabeledArray':
        return LabeledArray(
            data=np.ones(shape=tuple(shape.values()), dtype=dtype),
            axis_names=tuple(shape.keys()),
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
            for k in a.shape:
                if k in shape:
                    shape[k] = max(shape[k], a.shape[k])
                else:
                    shape[k] = a.shape[k]
        return shape

    def shape_broadcasted(self, *arrs: 'LabeledArray'):
        return self.broadcast_shapes(self, *arrs)

    def _data_aligned(self, shape: typ.Dict[str, int]) -> numpy.typing.ArrayLike:
        ndim_missing = len(shape) - np.ndim(self.data)
        data = np.expand_dims(self.data, tuple(~np.arange(ndim_missing)))
        # data = self.data[(..., ) + (np.newaxis, ) * ndim_missing]
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
            start: typ.Optional[int] = None,
            *args,
            **kwargs,
    ) -> 'LabeledArray':
        return LabeledArray(
            data=np.arange(start=start, *args, **kwargs),
            axis_names=(axis, )
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
            axis_names=tuple(shape.keys()),
        )

    def __array_ufunc__(
            self,
            function: np.ufunc,
            method: str,
            *inputs: 'LabeledArray',
            **kwargs: typ.Any,
    ):
        shape = self.broadcast_shapes(*inputs)
        inputs = [inp._data_aligned(shape) for inp in inputs]

        for inp in inputs:
            if hasattr(inp, '__array_ufunc__'):
                result = inp.__array_ufunc__(function, method, *inputs, **kwargs)
                if result is not NotImplemented:
                    return LabeledArray(
                        data=result,
                        axis_names=tuple(shape.keys()),
                    )

    def __array_function__(
            self,
            function: typ.Callable,
            types: typ.Collection,
            args: typ.Sequence,
            kwargs: typ.Dict[str, typ.Any],
    ):
        if function in [np.broadcast_to, np.broadcast_arrays]:
            raise ValueError('Broadcasting is not needed for LabeledArrays')
        elif function is np.result_type:
            return type(self)
        elif function in [
            np.ndim,
            np.min,
            np.nanmin,
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
        ]:

            result_axis_names = list(self.axis_names)
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
                    kwargs['axis'] = self.axis_names.index(kwargs['axis'])
                else:
                    kwargs['axis'] = tuple(self.axis_names.index(ax) for ax in kwargs['axis'])

            args = tuple(getattr(arg, 'data', arg) for arg in args)
            kwargs = {kw: getattr(kwargs[kw], 'data', kwargs[kw]) for kw in kwargs}
            types = tuple(type(arg) for arg in args if getattr(arg, '__array_function__', None) is not None)

            return type(self)(
                data=self.data.__array_function__(function, types, args, kwargs),
                axis_names=tuple(result_axis_names),
            )


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



GridType = typ.Dict[str, numpy.typing.ArrayLike]


@dataclasses.dataclass
class DataArray(kgpy.mixin.Copyable):

    data: numpy.typing.ArrayLike
    grid: typ.Dict[str, typ.Optional[numpy.typing.ArrayLike]]

    @property
    def grid_normalized(self) -> GridType:
        shape = np.broadcast(self.data, *self.grid.values()).shape
        grid_normalized = dict()
        for axis_name in self.grid:
            if self.grid[axis_name] is None:
                axis_index = self.axis_name_to_index(axis_name)
                axes_new = list(range(len(shape)))
                axes_new.remove(axis_index)
                grid_normalized[axis_name] = np.expand_dims(
                    a=np.arange(shape[axis_index]),
                    axis=axes_new,
                )
            else:
                grid_normalized[axis_name] = self.grid[axis_name]
        return grid_normalized

    @property
    def shape_tuple(self):
        return np.broadcast(self.data, *self.grid.values()).shape

    @property
    def shape(self) -> typ.Dict[str, int]:
        shape_tuple = self.shape_tuple
        shape = dict()
        for i, axis_name in enumerate(self.grid):
            shape[axis_name] = shape_tuple[i]
        return shape

    @property
    def data_broadcasted(self) -> numpy.typing.ArrayLike:
        return np.broadcast_to(self.data, self.shape_tuple, subok=True)

    @property
    def grid_broadcasted(self) -> typ.Dict[str, typ.Optional[numpy.typing.ArrayLike]]:
        shape = self.shape_tuple
        grid = self.grid.copy()
        for key in grid:
            grid[key] = np.broadcast_to(grid[key], shape=shape, subok=True)
        return grid

    @property
    def ndim(self):
        return self.data_broadcasted.ndim

    @property
    def size(self):
        return self.data_broadcasted.size

    def axis_name_to_index(self, key: str):
        return list(self.grid.keys()).index(key)

    def get_item(
            self,
            **key: typ.Union[int, slice, numpy.typing.ArrayLike]
    ) -> 'DataArray':

        result = DataArray(
            data=self.data_broadcasted,
            grid=self.grid_broadcasted
        )

        for axis_name in key:
            axis_index = self.axis_name_to_index(axis_name)
            indices = key[axis_name]

            result.data = take(a=result.data, key=indices, axis=axis_index)
            if np.isscalar(indices):
                del result.grid[axis_name]
            for k in result.grid:
                result.grid[k] = take(a=result.grid[k], key=indices, axis=axis_index)

        return result

    def set_item(
            self,
            value: numpy.typing.ArrayLike,
            **key: typ.Union[int, slice, numpy.typing.ArrayLike]
    ) -> typ.NoReturn:
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

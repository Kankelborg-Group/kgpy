import typing as typ
import abc
import dataclasses
import copy
import numpy as np
import numpy.typing
import astropy.units as u
import kgpy.mixin

__all__ = [
    'AbstractArray',
    'Array',
    'Range',
    'LinearSpace',
    'UniformRandomSpace',
]

AbstractArrayT = typ.TypeVar('AbstractArrayT', bound='AbstractLabeledArray')
OtherAbstractArrayT = typ.TypeVar('OtherAbstractArrayT', bound='AbstractLabeledArray')
ArrayT = typ.TypeVar('ArrayT', bound='Array')
RangeT = typ.TypeVar('RangeT', bound='Range')
LinearSpaceT = typ.TypeVar('LinearSpaceT', bound='LinearSpace')
UniformRandomSpaceT = typ.TypeVar('UniformRandomSpaceT', bound='UniformRandomSpace')


@dataclasses.dataclass(eq=False)
class AbstractArray(
    kgpy.mixin.Copyable,
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
):

    @property
    @abc.abstractmethod
    def value(self: AbstractArrayT) -> u.Quantity:
        pass

    @property
    @abc.abstractmethod
    def axes(self: AbstractArrayT) -> typ.List[str]:
        pass

    @property
    def shape(self: AbstractArrayT) -> typ.Dict[str, int]:
        shape = dict()
        for i in range(np.ndim(self.value)):
            shape[self.axes[i]] = self.value.shape[i]
        return shape

    @property
    def ndim(self: AbstractArrayT) -> int:
        return len(self.shape)

    @classmethod
    def broadcast_shapes(cls: typ.Type[AbstractArrayT], *arrs: AbstractArrayT) -> typ.Dict[str, int]:
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

    def shape_broadcasted(self: AbstractArrayT, *arrs: AbstractArrayT) -> typ.Dict[str, int]:
        return self.broadcast_shapes(self, *arrs)

    def _data_aligned(self: AbstractArrayT, shape: typ.Dict[str, int]) -> u.Quantity:
        ndim_missing = len(shape) - np.ndim(self.value)
        value = np.expand_dims(self.value, tuple(~np.arange(ndim_missing)))
        source = []
        destination = []
        for axis_index, axis_name in enumerate(self.axes):
            source.append(axis_index)
            destination.append(list(shape.keys()).index(axis_name))
        value = np.moveaxis(a=value, source=source, destination=destination)
        return value

    def add_axes(self: AbstractArrayT, axes: typ.List) -> AbstractArrayT:
        shape_new = {axis: 1 for axis in axes}
        shape = {**self.shape, **shape_new}
        return Array(
            value=self._data_aligned(shape),
            axes=list(shape.keys()),
        )

    def combine_axes(
            self: AbstractArrayT,
            axes: typ.Sequence[str],
            axis_new: typ.Optional[str] = None,
    ) -> 'Array':

        if axis_new is None:
            axis_new = ''.join(axes)

        axes_new = self.axes.copy()
        shape_new = self.shape
        for axis in axes:
            axes_new.append(axes_new.pop(axes_new.index(axis)))
            shape_new[axis] = shape_new.pop(axis)

        source = []
        destination = []
        for axis in axes:
            source.append(self.axes.index(axis))
            destination.append(axes_new.index(axis))

        for axis in axes:
            axes_new.remove(axis)
            shape_new.pop(axis)
        axes_new.append(axis_new)
        shape_new[axis_new] = -1

        return Array(
            value=np.moveaxis(self.value, source=source, destination=destination).reshape(tuple(shape_new.values())),
            axes=axes_new,
        )

    def matrix_multiply(
            self: AbstractArrayT,
            other: OtherAbstractArrayT,
            axis_rows: str,
            axis_columns: str,
    ) -> 'Array':

        shape = self.shape_broadcasted(other)
        shape_rows = shape.pop(axis_rows)
        shape_columns = shape.pop(axis_columns)
        shape = {**shape, axis_rows: shape_rows, axis_columns: shape_columns}

        data_self = self._data_aligned(shape)
        data_other = other._data_aligned(shape)

        return Array(
            value=np.matmul(data_self, data_other),
            axes=list(shape.keys()),
        )

    def matrix_determinant(
            self: AbstractArrayT,
            axis_rows: str,
            axis_columns: str
    ) -> 'Array':
        shape = self.shape
        if shape[axis_rows] != shape[axis_columns]:
            raise ValueError('Matrix must be square')

        if shape[axis_rows] == 2:
            a = self[{axis_rows: 0, axis_columns: 0}]
            b = self[{axis_rows: 0, axis_columns: 1}]
            c = self[{axis_rows: 1, axis_columns: 0}]
            d = self[{axis_rows: 1, axis_columns: 1}]
            return a * d - b * c

        elif shape[axis_rows] == 3:
            a = self[{axis_rows: 0, axis_columns: 0}]
            b = self[{axis_rows: 0, axis_columns: 1}]
            c = self[{axis_rows: 0, axis_columns: 2}]
            d = self[{axis_rows: 1, axis_columns: 0}]
            e = self[{axis_rows: 1, axis_columns: 1}]
            f = self[{axis_rows: 1, axis_columns: 2}]
            g = self[{axis_rows: 2, axis_columns: 0}]
            h = self[{axis_rows: 2, axis_columns: 1}]
            i = self[{axis_rows: 2, axis_columns: 2}]
            return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h)

        else:
            value = np.moveaxis(
                a=self.value,
                source=[self.axis_names.index(axis_rows), self.axis_names.index(axis_columns)],
                destination=[~1, ~0],
            )

            axes_new = self.axes.copy()
            axes_new.pop(axis_rows)
            axes_new.pop(axis_columns)

            return Array(
                value=np.linalg.det(value),
                axes=axes_new,
            )

    def matrix_inverse(
            self: AbstractArrayT,
            axis_rows: str,
            axis_columns: str,
    ) -> 'Array':
        shape = self.shape
        if shape[axis_rows] != shape[axis_columns]:
            raise ValueError('Matrix must be square')

        if shape[axis_rows] == 2:
            result = Array(value=self.value.copy(), axes=self.axes.copy())
            result[{axis_rows: 0, axis_columns: 0}] = self[{axis_rows: 1, axis_columns: 1}]
            result[{axis_rows: 1, axis_columns: 1}] = self[{axis_rows: 0, axis_columns: 0}]
            result[{axis_rows: 0, axis_columns: 1}] = -self[{axis_rows: 0, axis_columns: 1}]
            result[{axis_rows: 1, axis_columns: 0}] = -self[{axis_rows: 1, axis_columns: 0}]
            return result / self.matrix_determinant(axis_rows=axis_rows, axis_columns=axis_columns)

        elif shape[axis_rows] == 3:
            a = self[{axis_rows: 0, axis_columns: 0}]
            b = self[{axis_rows: 0, axis_columns: 1}]
            c = self[{axis_rows: 0, axis_columns: 2}]
            d = self[{axis_rows: 1, axis_columns: 0}]
            e = self[{axis_rows: 1, axis_columns: 1}]
            f = self[{axis_rows: 1, axis_columns: 2}]
            g = self[{axis_rows: 2, axis_columns: 0}]
            h = self[{axis_rows: 2, axis_columns: 1}]
            i = self[{axis_rows: 2, axis_columns: 2}]

            result = Array(value=self.value.copy(), axes=self.axes.copy())
            result[{axis_rows: 0, axis_columns: 0}] = (e * i - f * h)
            result[{axis_rows: 0, axis_columns: 1}] = -(b * i - c * h)
            result[{axis_rows: 0, axis_columns: 2}] = (b * f - c * e)
            result[{axis_rows: 1, axis_columns: 0}] = -(d * i - f * g)
            result[{axis_rows: 1, axis_columns: 1}] = (a * i - c * g)
            result[{axis_rows: 1, axis_columns: 2}] = -(a * f - c * d)
            result[{axis_rows: 2, axis_columns: 0}] = (d * h - e * g)
            result[{axis_rows: 2, axis_columns: 1}] = -(a * h - b * g)
            result[{axis_rows: 2, axis_columns: 2}] = (a * e - b * d)
            return result / self.matrix_determinant(axis_rows=axis_rows, axis_columns=axis_columns)

        else:
            index_axis_rows = self.axes.index(axis_rows)
            index_axis_columns = self.axes.index(axis_columns)
            value = np.moveaxis(
                a=self.value,
                source=[index_axis_rows, index_axis_columns],
                destination=[~1, ~0],
            )

            axes_new = self.axes.copy()
            axes_new.append(axes_new.pop(index_axis_rows))
            axes_new.append(axes_new.pop(index_axis_columns))

            return Array(
                value=np.linalg.inv(value),
                axes=axes_new,
            )

    def __mul__(self: AbstractArrayT, other: typ.Union['ArrayLike', u.Unit]):
        if isinstance(other, u.Unit):
            return Array(
                value=self.value * other,
                axes=self.axes.copy(),
            )
        else:
            return super().__mul__(other)

    def __array_ufunc__(
            self: AbstractArrayT,
            function: np.ufunc,
            method: str,
            *inputs: typ.Union[u.Quantity, 'Array'],
            **kwargs: typ.Any,
    ) -> 'Array':
        inputs = [Array(value=inp, axes=[]) if np.isscalar(inp) else inp for inp in inputs]
        for inp in inputs:
            if not isinstance(inp, AbstractArray):
                name = f'{AbstractArray.__module__}.{AbstractArray.__qualname__}'
                raise ValueError(f'Inputs must be scalars or instances of {name}')

        shape = self.broadcast_shapes(*inputs)
        inputs = [inp._data_aligned(shape) for inp in inputs]

        for inp in inputs:
            if hasattr(inp, '__array_ufunc__'):
                result = inp.__array_ufunc__(function, method, *inputs, **kwargs)
                if result is not NotImplemented:
                    return Array(
                        value=result,
                        axes=list(shape.keys()),
                    )
        raise ValueError

    def __array_function__(
            self: AbstractArrayT,
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
            return Array(
                value=np.broadcast_to(array=array._data_aligned(shape), shape=tuple(shape.values()), subok=subok),
                axes=list(shape.keys()),
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

            result_value = np.unravel_index(indices=indices, shape=tuple(shape.values()))
            result = dict()
            for axis, value in zip(shape, result_value):
                result[axis] = Array(
                    data=value,
                    axes=self.axes.copy(),
                )
            return result

        elif function is np.linalg.inv:
            raise ValueError(f'{function} is unsupported, use kgpy.LabeledArray.matrix_inverse() instead.')

        elif function is np.stack:
            if 'arrays' in kwargs:
                arrays = kwargs.pop('arrays')
            else:
                arrays = args[0]    # type: typ.List[Array]

            if 'axis' in kwargs:
                axis = kwargs.pop('axis')
            else:
                raise ValueError('axis must be specified')

            shape = self.broadcast_shapes(*arrays)
            arrays = [arr._data_aligned(shape) for arr in arrays]

            return Array(
                value=np.stack(arrays=arrays, axis=0, **kwargs),
                axes=[axis] + self.axes,
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

            labeled_arrays = [arg for arg in args if isinstance(arg, Array)]
            labeled_arrays += [kwargs[k] for k in kwargs if isinstance(kwargs[k], Array)]
            shape = Array.broadcast_shapes(*labeled_arrays)
            axes = list(shape.keys())

            args = tuple(arg._data_aligned(shape) if isinstance(arg, Array) else arg for arg in args)
            kwargs = {k: kwargs[k]._data_aligned(shape) if isinstance(kwargs[k], Array) else kwargs[k] for k in kwargs}
            types = tuple(type(arg) for arg in args if getattr(arg, '__array_function__', None) is not None)

            axes_new = axes.copy()
            if function is not np.isclose:
                if 'keepdims' not in kwargs:
                    if 'axis' in kwargs:
                        if np.isscalar(kwargs['axis']):
                            axes_new.remove(kwargs['axis'])
                        else:
                            for axis in kwargs['axis']:
                                axes_new.remove(axis)
                    else:
                        axes_new = []

            if 'axis' in kwargs:
                if np.isscalar(kwargs['axis']):
                    kwargs['axis'] = axes.index(kwargs['axis'])
                else:
                    kwargs['axis'] = tuple(axes.index(ax) for ax in kwargs['axis'])

            value = self.value.__array_function__(function, types, args, kwargs)

            if function in [
                np.array_equal,
            ]:
                return value
            else:
                return type(self)(
                    value=value,
                    axes=axes_new,
                )
        else:
            raise ValueError('Unsupported function')

    def __bool__(self: AbstractArrayT) -> bool:
        return self.value.__bool__()

    def __getitem__(
            self: AbstractArrayT,
            item: typ.Union[typ.Dict[str, typ.Union[int, slice, 'Array']], 'Array'],
    ) -> 'Array':

        if isinstance(item, Array):
            value = np.moveaxis(
                a=self.value,
                source=[self.axes.index(axis) for axis in item.axes],
                destination=np.arange(len(item.axes)),
            )

            return Array(
                value=np.moveaxis(value[item.value], 0, ~0),
                axes=[axis for axis in self.axes if axis not in item.axes] + ['boolean']
            )

        else:
            axes_advanced = []
            axes_indices_advanced = []
            item_advanced = dict()
            for axis in item:
                if isinstance(item[axis], Array):
                    axes_advanced.append(axis)
                    axes_indices_advanced.append(self.axes.index(axis))
                    item_advanced[axis] = item[axis]

            shape_advanced = self.broadcast_shapes(*item_advanced.values())

            value = np.moveaxis(
                a=self.value,
                source=axes_indices_advanced,
                destination=list(range(len(axes_indices_advanced))),
            )

            axes = self.axes.copy()
            for a, axis in enumerate(axes_advanced):
                axes.remove(axis)
                axes.insert(a, axis)

            axes_new = axes.copy()
            index = [slice(None)] * self.ndim
            for axis_name in item:
                item_axis = item[axis_name]
                if isinstance(item_axis, Array):
                    item_axis = item_axis._data_aligned(shape_advanced)
                index[axes.index(axis_name)] = item_axis
                if not isinstance(item_axis, slice):
                    axes_new.remove(axis_name)

            return Array(
                value=value[tuple(index)],
                axes=list(shape_advanced.keys()) + axes_new,
            )

    @classmethod
    def ndindex(cls: typ.Type[AbstractArrayT], shape: typ.Dict[str, int]) -> typ.Iterator[typ.Dict[str, int]]:
        shape_tuple = tuple(shape.values())
        for index in np.ndindex(shape_tuple):
            yield dict(zip(shape.keys(), index))


ArrayLike = typ.Union[float, u.Quantity, AbstractArray]


@dataclasses.dataclass(eq=False)
class Array(
    AbstractArray,
):
    value: u.Quantity = 0 * u.dimensionless_unscaled
    axes: typ.Optional[typ.List[str]] = None

    def __post_init__(self: ArrayT):
        if self.axes is None:
            self.axes = []
        if np.ndim(self.value) != len(self.axes):
            raise ValueError('The number of axis names must match the number of dimensions.')
        if len(self.axes) != len(set(self.axes)):
            raise ValueError('Each axis name must be unique.')

    @classmethod
    def empty(cls: typ.Type[ArrayT], shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> ArrayT:
        return cls(
            value=np.empty(shape=tuple(shape.values()), dtype=dtype),
            axes=list(shape.keys()),
        )

    @classmethod
    def zeros(cls: typ.Type[ArrayT], shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> ArrayT:
        return cls(
            value=np.zeros(shape=tuple(shape.values()), dtype=dtype),
            axes=list(shape.keys()),
        )

    @classmethod
    def ones(cls: typ.Type[ArrayT], shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> ArrayT:
        return cls(
            value=np.ones(shape=tuple(shape.values()), dtype=dtype),
            axes=list(shape.keys()),
        )

    def __setitem__(
            self: ArrayT,
            key: typ.Union[typ.Dict[str, typ.Union[int, slice, ArrayT]], ArrayT],
            value: typ.Union[float, ArrayT],
    ) -> typ.NoReturn:

        if isinstance(key, Array):
            shape = self.shape_broadcasted(key)
            self._data_aligned(shape)[key._data_aligned(shape)] = value

        else:
            index = [slice(None)] * self.ndim
            axes = self.axes.copy()
            for axis in key:
                item_axis = key[axis]
                if isinstance(item_axis, int):
                    axes.remove(axis)
                if isinstance(item_axis, Array):
                    item_axis = item_axis._data_aligned(self.shape_broadcasted(item_axis))
                index[self.axes.index(axis)] = item_axis

            self.value[tuple(index)] = value._data_aligned({axis: None for axis in axes})

    def view(self) -> 'Array':
        other = super().view()      # type: Array
        other.data = self.data
        other.axis_names = self.axis_names
        return other

    def copy(self) -> 'Array':
        other = super().copy()      # type: Array
        other.data = self.data.copy()
        other.axis_names = copy.deepcopy(self.axis_names)
        return other


@dataclasses.dataclass
class Range(AbstractArray):

    start: int = 0
    stop: int = None
    step: int = 1
    axis: str = None

    @property
    def value(self: RangeT) -> numpy.ndarray:
        return np.arange(
            start=self.start,
            stop=self.stop,
            step=self.step,
        )

    @property
    def axes(self: RangeT) -> typ.List[str]:
        return [self.axis]

    def view(self: RangeT) -> RangeT:
        other = super().view()
        other.start = self.start
        other.stop = self.stop
        other.step = self.step
        other.axis = self.axis
        return other

    def copy(self: RangeT) -> RangeT:
        other = super().copy()
        other.start = self.start
        other.stop = self.stop
        other.step = self.step
        other.axis = self.axis
        return other


StartArrayT = typ.TypeVar('StartArrayT', bound=AbstractArray)
StopArrayT = typ.TypeVar('StopArrayT', bound=AbstractArray)


@dataclasses.dataclass(eq=False)
class LinearSpace(
    AbstractArray,
    typ.Generic[StartArrayT, StopArrayT],
):
    start: typ.Union[u.Quantity, StartArrayT] = None
    stop: typ.Union[u.Quantity, StopArrayT] = None
    num: int = None
    axis: str = None

    @property
    def _start_normalized(self: LinearSpaceT) -> StartArrayT:
        if not isinstance(self.start, AbstractArray):
            return Array(self.start)
        else:
            return self.start

    @property
    def _stop_normalized(self: LinearSpaceT) -> StartArrayT:
        if not isinstance(self.stop, AbstractArray):
            return Array(self.stop)
        else:
            return self.stop

    @property
    def start_broadcasted(self: LinearSpaceT) -> StartArrayT:
        return np.broadcast_to(self._start_normalized, shape=self.shape, subok=True)

    @property
    def stop_broadcasted(self):
        return np.broadcast_to(self._stop_normalized, shape=self.shape, subok=True)

    @property
    def range(self: LinearSpaceT) -> Array:
        return self.stop - self.start

    @property
    def shape(self: LinearSpaceT) -> typ.Dict[str, int]:
        shape = self.broadcast_shapes(self._start_normalized, self._stop_normalized)
        # shape = self.start.shape_broadcasted(self.stop)
        # if self.axis in shape:
        #     raise ValueError('Axis already defined, pick a new axis.')
        shape[self.axis] = self.num
        return shape

    @property
    def value(self: LinearSpaceT) -> u.Quantity:
        shape = self.shape
        return np.linspace(
            start=self._start_normalized._data_aligned(shape)[..., 0],
            stop=self._stop_normalized._data_aligned(shape)[..., 0],
            num=self.num,
            axis=~0,
        )

    @property
    def axes(self: LinearSpaceT) -> typ.List[str]:
        return list(self.shape.keys())

    def view(self: LinearSpaceT) -> LinearSpaceT:
        other = super().view()
        other.start = self.start
        other.stop = self.stop
        other.num = self.num
        other.axis = self.axis
        return other

    def copy(self: LinearSpaceT) -> LinearSpaceT:
        other = super().copy()
        other.start = self.start.copy()
        other.stop = self.stop.copy()
        other.num = self.num
        other.axis = self.axis
        return other


@dataclasses.dataclass
class UniformRandomSpace(LinearSpace):

    seed: int = 42

    @property
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=self.seed)

    @property
    def value(self: UniformRandomSpaceT):
        shape = self.shape
        return self._rng.uniform(
            low=self.start_broadcasted._data_aligned(shape),
            high=self.stop_broadcasted._data_aligned(shape),
        )

    def view(self: UniformRandomSpaceT) -> UniformRandomSpaceT:
        other = super().view()
        other.seed = self.seed
        return other

    def copy(self: UniformRandomSpaceT) -> UniformRandomSpaceT:
        other = super().copy()
        other.seed = self.seed
        return other
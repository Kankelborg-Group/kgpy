import typing as typ
import abc
import dataclasses
import copy
import numpy as np
import numpy.typing
import astropy.units as u
import kgpy.mixin
import kgpy.units

__all__ = [
    'AbstractArray',
    'Array',
    'Range',
    'LinearSpace',
    'UniformRandomSpace',
]

NDArrayMethodsMixinT = typ.TypeVar('NDArrayMethodsMixinT', bound='NDArrayMethodsMixin')
ArrT = typ.TypeVar('ArrT', bound=kgpy.units.QuantityLike)
ArrayInterfaceT = typ.TypeVar('ArrayInterfaceT', bound='ArrayInterface')
AbstractArrayT = typ.TypeVar('AbstractArrayT', bound='AbstractArray')
OtherAbstractArrayT = typ.TypeVar('OtherAbstractArrayT', bound='AbstractArray')
ArrayT = typ.TypeVar('ArrayT', bound='Array')
RangeT = typ.TypeVar('RangeT', bound='Range')
_SpaceMixinT = typ.TypeVar('_SpaceMixinT', bound='_SpaceMixin')
StartArrayT = typ.TypeVar('StartArrayT', bound='ArrayLike')
StopArrayT = typ.TypeVar('StopArrayT', bound='ArrayLike')
_LinearMixinT = typ.TypeVar('_LinearMixinT', bound='_LinearMixin')
LinearSpaceT = typ.TypeVar('LinearSpaceT', bound='LinearSpace')
_RandomSpaceMixinT = typ.TypeVar('_RandomSpaceMixinT', bound='_RandomSpaceMixin')
UniformRandomSpaceT = typ.TypeVar('UniformRandomSpaceT', bound='UniformRandomSpace')
CenterT = typ.TypeVar('CenterT', bound='ArrayLike')
WidthT = typ.TypeVar('WidthT', bound='ArrayLike')
_NormalMixinT = typ.TypeVar('_NormalMixinT', bound='WidthMixin')
NormalRandomSpaceT = typ.TypeVar('NormalRandomSpaceT', bound='NormalRandomSpace')


@dataclasses.dataclass(eq=False)
class NDArrayMethodsMixin:

    def broadcast_to(
            self: NDArrayMethodsMixinT,
            shape: typ.Dict[str, int],
    ) -> NDArrayMethodsMixinT:
        return np.broadcast_to(self, shape=shape)

    def min(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.min(self, axis=axis, where=where)

    def max(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.max(self, axis=axis, where=where)

    def sum(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.sum(self, axis=axis, where=where)

    def ptp(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
    ) -> NDArrayMethodsMixinT:
        return np.ptp(self, axis=axis)

    def mean(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.mean(self, axis=axis, where=where)

    def all(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.all(self, axis=axis, where=where)

    def any(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.any(self, axis=axis, where=where)


@dataclasses.dataclass(eq=False)
class ArrayInterface(
    kgpy.mixin.Copyable,
    NDArrayMethodsMixin,
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
):

    @property
    @abc.abstractmethod
    def shape(self: ArrayInterfaceT) -> typ.Dict[str, int]:
        pass

    @property
    def broadcasted(self: ArrayInterfaceT) -> ArrayInterfaceT:
        return np.broadcast_to(self, shape=self.shape)

    def ndindex(
            self: typ.Type[ArrayInterfaceT],
            axis_ignored: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
    ) -> typ.Iterator[typ.Dict[str, int]]:

        if axis_ignored is None:
            axis_ignored = []
        elif isinstance(axis_ignored, str):
            axis_ignored = [axis_ignored]

        shape = self.shape
        for axis in axis_ignored:
            shape.pop(axis)
        shape_tuple = tuple(shape.values())
        for index in np.ndindex(*shape_tuple):
            yield dict(zip(shape.keys(), index))

    @abc.abstractmethod
    def combine_axes(
            self: ArrayInterfaceT,
            axes: typ.Sequence[str],
            axis_new: typ.Optional[str] = None,
    ) -> ArrayInterfaceT:
        pass


@dataclasses.dataclass(eq=False)
class AbstractArray(
    ArrayInterface,
    typ.Generic[ArrT],
):

    type_array_primary: typ.ClassVar[typ.Type] = np.ndarray
    type_array_auxiliary: typ.ClassVar[typ.Tuple[typ.Type, ...]] = (bool, int, float, complex, np.generic)
    type_array: typ.ClassVar[typ.Tuple[typ.Type, ...]] = type_array_auxiliary + (type_array_primary, )

    @property
    @abc.abstractmethod
    def array(self: AbstractArrayT) -> ArrT:
        pass

    @property
    def _array_normalized(self: AbstractArrayT) -> np.ndarray:
        value = self.array
        if isinstance(value, self.type_array_auxiliary):
            value = np.array(value)
        return value

    @property
    @abc.abstractmethod
    def unit(self) -> typ.Union[float, u.Unit]:
        return 1

    @property
    @abc.abstractmethod
    def axes(self: AbstractArrayT) -> typ.Optional[typ.List[str]]:
        pass

    @property
    def _axes_normalized(self: AbstractArrayT) -> typ.List[str]:
        if self.axes is not None:
            return self.axes
        else:
            return []

    @property
    @abc.abstractmethod
    def shape(self: AbstractArrayT) -> typ.Dict[str, int]:
        return dict()

    @property
    def ndim(self: AbstractArrayT) -> int:
        return len(self.shape)

    @classmethod
    def broadcast_shapes(cls: typ.Type[AbstractArrayT], *arrs: AbstractArrayT) -> typ.Dict[str, int]:
        shape = dict()      # type: typ.Dict[str, int]
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

    def array_aligned(self: AbstractArrayT, shape: typ.Dict[str, int]) -> ArrT:
        ndim_missing = len(shape) - np.ndim(self.array)
        value = np.expand_dims(self.array, tuple(~np.arange(ndim_missing)))
        source = []
        destination = []
        for axis_index, axis_name in enumerate(self._axes_normalized):
            source.append(axis_index)
            destination.append(list(shape.keys()).index(axis_name))
        value = np.moveaxis(a=value, source=source, destination=destination)
        return value

    def add_axes(self: AbstractArrayT, axes: typ.List) -> AbstractArrayT:
        shape_new = {axis: 1 for axis in axes}
        shape = {**self.shape, **shape_new}
        return Array(
            array=self.array_aligned(shape),
            axes=list(shape.keys()),
        )

    def combine_axes(
            self: AbstractArrayT,
            axes: typ.Sequence[str],
            axis_new: typ.Optional[str] = None,
    ) -> 'Array':

        if axis_new is None:
            axis_new = ''.join(axes)

        axes_new = self._axes_normalized.copy()
        shape_new = self.shape
        for axis in axes:
            axes_new.append(axes_new.pop(axes_new.index(axis)))
            shape_new[axis] = shape_new.pop(axis)

        source = []
        destination = []
        for axis in axes:
            source.append(self._axes_normalized.index(axis))
            destination.append(axes_new.index(axis))

        for axis in axes:
            axes_new.remove(axis)
            shape_new.pop(axis)
        axes_new.append(axis_new)
        shape_new[axis_new] = -1

        return Array(
            array=np.moveaxis(self.array, source=source, destination=destination).reshape(tuple(shape_new.values())),
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

        data_self = self.array_aligned(shape)
        data_other = other.array_aligned(shape)

        return Array(
            array=np.matmul(data_self, data_other),
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
                a=self.array,
                source=[self.axis_names.index(axis_rows), self.axis_names.index(axis_columns)],
                destination=[~1, ~0],
            )

            axes_new = self._axes_normalized.copy()
            axes_new.remove(axis_rows)
            axes_new.remove(axis_columns)

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
            result = Array(array=self.array.copy(), axes=self._axes_normalized.copy())
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

            result = Array(array=self.array.copy(), axes=self._axes_normalized.copy())
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
            index_axis_rows = self._axes_normalized.index(axis_rows)
            index_axis_columns = self._axes_normalized.index(axis_columns)
            value = np.moveaxis(
                a=self.array,
                source=[index_axis_rows, index_axis_columns],
                destination=[~1, ~0],
            )

            axes_new = self._axes_normalized.copy()
            axes_new.remove(axis_rows)
            axes_new.remove(axis_columns)
            axes_new.append(axis_rows)
            axes_new.append(axis_columns)

            return Array(
                array=np.linalg.inv(value),
                axes=axes_new,
            )

    def __mul__(self: AbstractArrayT, other: typ.Union['ArrayLike', u.UnitBase]):
        if isinstance(other, u.UnitBase):
            return Array(
                array=self.array * other,
                axes=self._axes_normalized.copy(),
            )
        else:
            return super().__mul__(other)

    def __lshift__(self, other: u.UnitBase) -> 'Array':
        axes = self.axes
        if axes is not None:
            axes = axes.copy()
        return Array(
            array=self.array << other,
            axes=axes
        )

    def __array_ufunc__(
            self,
            function,
            method,
            *inputs,
            **kwargs,
    ) -> 'Array':

        inputs_normalized = []

        for inp in inputs:
            if isinstance(inp, self.type_array):
                inp = Array(inp)
            elif isinstance(inp, AbstractArray):
                pass
            else:
                return NotImplemented
            inputs_normalized.append(inp)
        inputs = inputs_normalized

        shape = self.broadcast_shapes(*inputs)
        inputs = tuple(inp.array_aligned(shape) for inp in inputs)

        for inp in inputs:
            result = inp.__array_ufunc__(function, method, *inputs, **kwargs)
            if result is not NotImplemented:
                return Array(
                    array=result,
                    axes=list(shape.keys()),
                )

        return NotImplemented

    def __array_function__(
            self: AbstractArrayT,
            func: typ.Callable,
            types: typ.Collection,
            args: typ.Tuple,
            kwargs: typ.Dict[str, typ.Any],
    ):
        if func is np.broadcast_to:
            args = list(args)
            if 'array' in kwargs:
                array = kwargs.pop('array')
            else:
                array = args.pop(0)

            if 'shape' in kwargs:
                shape = kwargs.pop('shape')
            else:
                shape = args.pop(0)

            return Array(
                array=np.broadcast_to(array.array_aligned(shape), tuple(shape.values()), subok=True),
                axes=list(shape.keys()),
            )
        elif func is np.result_type:
            return type(self)
        elif func is np.unravel_index:
            args = list(args)

            if args:
                indices = args.pop(0)
            else:
                indices = kwargs.pop('indices')

            if args:
                shape = args.pop(0)
            else:
                shape = kwargs.pop('shape')

            result_value = np.unravel_index(indices=indices.array, shape=tuple(shape.values()))
            result = dict()     # type: typ.Dict[str, Array]
            for axis, array in zip(shape, result_value):
                result[axis] = Array(
                    array=array,
                    axes=self._axes_normalized.copy(),
                )
            return result

        elif func is np.linalg.inv:
            raise ValueError(f'{func} is unsupported, use kgpy.LabeledArray.matrix_inverse() instead.')

        elif func is np.stack:
            args = list(args)
            kwargs = kwargs.copy()

            if args:
                if 'arrays' in kwargs:
                    raise TypeError(f"{func} got multiple values for 'arrays'")
                arrays = args.pop(0)
            else:
                arrays = kwargs.pop('arrays')

            if args:
                if 'axis' in kwargs:
                    raise TypeError(f"{func} got multiple values for 'axis'")
                axis = args.pop(0)
            else:
                axis = kwargs.pop('axis')

            shape = self.broadcast_shapes(*arrays)
            arrays = [Array(arr) if not isinstance(arr, ArrayInterface) else arr for arr in arrays]
            for array in arrays:
                if not isinstance(array, AbstractArray):
                    return NotImplemented
            arrays = [np.broadcast_to(arr, shape).array for arr in arrays]

            return Array(
                array=np.stack(arrays=arrays, axis=0, **kwargs),
                axes=[axis] + list(shape.keys()),
            )

        elif func is np.concatenate:
            args = list(args)
            if args:
                arrays = args.pop(0)
            else:
                arrays = kwargs['arrays']

            if args:
                axis = args.pop(0)
            else:
                axis = kwargs['axis']

            arrays = [Array(arr) if not isinstance(arr, ArrayInterface) else arr for arr in arrays]
            for arr in arrays:
                if not isinstance(arr, kgpy.labeled.AbstractArray):
                    return NotImplemented

            shape = self.broadcast_shapes(*arrays)
            arrays = [np.broadcast_to(arr, shape).array for arr in arrays]

            axes = list(shape.keys())
            return Array(
                array=func(arrays=arrays, axis=axes.index(axis)),
                axes=axes,
            )

        elif func in [
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
            np.roll,
        ]:

            labeled_arrays = [arg for arg in args if isinstance(arg, AbstractArray)]
            labeled_arrays += [kwargs[k] for k in kwargs if isinstance(kwargs[k], AbstractArray)]
            shape = Array.broadcast_shapes(*labeled_arrays)
            axes = list(shape.keys())

            args = tuple(arg.array_aligned(shape) if isinstance(arg, AbstractArray) else arg for arg in args)
            kwargs = {k: kwargs[k].array_aligned(shape) if isinstance(kwargs[k], AbstractArray) else kwargs[k] for k in kwargs}
            types = tuple(type(arg) for arg in args if getattr(arg, '__array_function__', None) is not None)

            axes_new = axes.copy()
            if func not in [np.isclose, np.roll]:
                if 'keepdims' not in kwargs:
                    if 'axis' in kwargs:
                        if kwargs['axis'] is None:
                            axes_new = []
                        elif np.isscalar(kwargs['axis']):
                            axes_new.remove(kwargs['axis'])
                        else:
                            for axis in kwargs['axis']:
                                axes_new.remove(axis)
                    else:
                        axes_new = []

            if 'axis' in kwargs:
                if kwargs['axis'] is None:
                    pass
                elif np.isscalar(kwargs['axis']):
                    kwargs['axis'] = axes.index(kwargs['axis'])
                else:
                    kwargs['axis'] = tuple(axes.index(ax) for ax in kwargs['axis'])

            array = self.array
            if not hasattr(array, '__array_function__'):
                array = np.array(array)
            array = array.__array_function__(func, types, args, kwargs)

            if func in [
                np.array_equal,
            ]:
                return array
            else:
                return Array(
                    array=array,
                    axes=axes_new,
                )
        else:
            raise ValueError('Unsupported function')

    def __bool__(self: AbstractArrayT) -> bool:
        return self.array.__bool__()

    # @typ.overload
    # def __getitem__(self: AbstractArrayT, item: typ.Dict[str, int]) -> 'Array': ...
    #
    # @typ.overload
    # def __getitem__(self: AbstractArrayT, item: typ.Dict[str, slice]) -> 'Array': ...
    #
    # @typ.overload
    # def __getitem__(self: AbstractArrayT, item: typ.Dict[str, AbstractArrayT]) -> 'Array': ...
    #
    # @typ.overload
    # def __getitem__(self: AbstractArrayT, item: 'AbstractArray') -> 'Array': ...

    def __getitem__(
            self: AbstractArrayT,
            item: typ.Union[typ.Dict[str, typ.Union[int, slice, 'AbstractArray']], 'AbstractArray'],
    ) -> 'Array':

        if isinstance(item, AbstractArray):
            value = np.moveaxis(
                a=self.array,
                source=[self._axes_normalized.index(axis) for axis in item._axes_normalized],
                destination=np.arange(len(item._axes_normalized)),
            )

            return Array(
                array=np.moveaxis(value[item.array], 0, ~0),
                axes=[axis for axis in self._axes_normalized if axis not in item._axes_normalized] + ['boolean']
            )

        else:
            item_casted = typ.cast(typ.Dict[str, typ.Union[int, slice, AbstractArray]], item)
            axes_advanced = []
            axes_indices_advanced = []
            item_advanced = dict()      # type: typ.Dict[str, AbstractArray]
            for axis in item_casted:
                item_axis = item_casted[axis]
                if isinstance(item_axis, AbstractArray):
                    axes_advanced.append(axis)
                    axes_indices_advanced.append(self._axes_normalized.index(axis))
                    item_advanced[axis] = item_axis

            shape_advanced = self.broadcast_shapes(*item_advanced.values())

            value = np.moveaxis(
                a=self.array,
                source=axes_indices_advanced,
                destination=list(range(len(axes_indices_advanced))),
            )

            axes = self._axes_normalized.copy()
            for a, axis in enumerate(axes_advanced):
                axes.remove(axis)
                axes.insert(a, axis)

            axes_new = axes.copy()
            index = [slice(None)] * self.ndim   # type: typ.List[typ.Union[int, slice, AbstractArray]]
            for axis_name in item_casted:
                item_axis = item_casted[axis_name]
                if isinstance(item_axis, AbstractArray):
                    item_axis = item_axis.array_aligned(shape_advanced)
                index[axes.index(axis_name)] = item_axis
                if not isinstance(item_axis, slice):
                    axes_new.remove(axis_name)

            return Array(
                array=value[tuple(index)],
                axes=list(shape_advanced.keys()) + axes_new,
            )
        # else:
        #     raise ValueError('Invalid index type')


ArrayLike = typ.Union[kgpy.units.QuantityLike, AbstractArray]


@dataclasses.dataclass(eq=False)
class Array(
    AbstractArray[ArrT],
):
    array: ArrT = 0 * u.dimensionless_unscaled
    axes: typ.Optional[typ.List[str]] = None

    def __post_init__(self: ArrayT):
        if self.axes is None:
            self.axes = []
        if np.ndim(self.array) != len(self.axes):
            raise ValueError('The number of axis names must match the number of dimensions.')
        if len(self.axes) != len(set(self.axes)):
            raise ValueError('Each axis name must be unique.')

    @classmethod
    def empty(cls: typ.Type[ArrayT], shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> ArrayT:
        return cls(
            array=np.empty(shape=tuple(shape.values()), dtype=dtype),
            axes=list(shape.keys()),
        )

    @classmethod
    def zeros(cls: typ.Type[ArrayT], shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> ArrayT:
        return cls(
            array=np.zeros(shape=tuple(shape.values()), dtype=dtype),
            axes=list(shape.keys()),
        )

    @classmethod
    def ones(cls: typ.Type[ArrayT], shape: typ.Dict[str, int], dtype: numpy.typing.DTypeLike = float) -> ArrayT:
        return cls(
            array=np.ones(shape=tuple(shape.values()), dtype=dtype),
            axes=list(shape.keys()),
        )

    @property
    def unit(self) -> typ.Union[float, u.Unit]:
        unit = super().unit
        if hasattr(self.array, 'unit'):
            unit = self.array.unit
        return unit

    @property
    def shape(self: ArrayT) -> typ.Dict[str, int]:
        shape = super().shape
        for i in range(np.ndim(self.array)):
            shape[self._axes_normalized[i]] = self.array.shape[i]
        return shape

    def __setitem__(
            self: ArrayT,
            key: typ.Union[typ.Dict[str, typ.Union[int, slice, AbstractArray]], AbstractArray],
            value: typ.Union[float, AbstractArray],
    ) -> None:

        if not isinstance(value, AbstractArray):
            value = Array(value)

        if isinstance(key, Array):
            shape = self.shape_broadcasted(key)
            self.array_aligned(shape)[key.array_aligned(shape)] = value.array

        else:
            key_casted = typ.cast(typ.Dict[str, typ.Union[int, slice, AbstractArray]], key)
            index = [slice(None)] * self.ndim   # type: typ.List[typ.Union[int, slice, AbstractArray]]
            axes = self._axes_normalized.copy()
            for axis in key_casted:
                item_axis = key_casted[axis]
                if isinstance(item_axis, int):
                    axes.remove(axis)
                if isinstance(item_axis, Array):
                    item_axis = item_axis.array_aligned(self.shape_broadcasted(item_axis))
                index[self._axes_normalized.index(axis)] = item_axis

            self.array[tuple(index)] = value.array_aligned({axis: 1 for axis in axes})


@dataclasses.dataclass(eq=False)
class Range(AbstractArray[np.ndarray]):

    start: int = 0
    stop: int = None
    step: int = 1
    axis: str = None

    @property
    def unit(self) -> typ.Union[float, u.Unit]:
        unit = super().unit
        if hasattr(self.start, 'unit'):
            unit = self.start.unit
        return unit

    @property
    def shape(self: RangeT) -> typ.Dict[str, int]:
        shape = super().shape
        for i in range(np.ndim(self.array)):
            shape[self._axes_normalized[i]] = self.array.shape[i]
        return shape

    @property
    def array(self: RangeT) -> np.ndarray:
        return np.arange(
            start=self.start,
            stop=self.stop,
            step=self.step,
        )

    @property
    def axes(self: RangeT) -> typ.List[str]:
        return [self.axis]


@dataclasses.dataclass(eq=False)
class _SpaceMixin(
    AbstractArray[kgpy.units.QuantityLike],
):
    num: int = None
    endpoint: bool = True
    axis: str = None

    @property
    def shape(self: _SpaceMixinT) -> typ.Dict[str, int]:
        shape = super().shape
        shape[self.axis] = self.num
        return shape

    @property
    def axes(self: _SpaceMixinT) -> typ.List[str]:
        return list(self.shape.keys())


@dataclasses.dataclass(eq=False)
class _LinearMixin(
    AbstractArray[kgpy.units.QuantityLike],
    typ.Generic[StartArrayT, StopArrayT],
):
    start: StartArrayT = None
    stop: StopArrayT = None

    @property
    def _start_normalized(self: _LinearMixinT) -> AbstractArray:
        if not isinstance(self.start, AbstractArray):
            return Array(self.start)
        else:
            return self.start

    @property
    def _stop_normalized(self: _LinearMixinT) -> AbstractArray:
        if not isinstance(self.stop, AbstractArray):
            return Array(self.stop)
        else:
            return self.stop

    @property
    def start_broadcasted(self: _LinearMixinT) -> AbstractArray:
        return np.broadcast_to(self._start_normalized, shape=self.shape, subok=True)

    @property
    def stop_broadcasted(self: _LinearMixinT) -> AbstractArray:
        return np.broadcast_to(self._stop_normalized, shape=self.shape, subok=True)

    @property
    def unit(self) -> typ.Union[float, u.Unit]:
        unit = super().unit
        if hasattr(self.start, 'unit'):
            unit = self.start.unit
        return unit

    @property
    def range(self: _LinearMixinT) -> Array:
        return self.stop - self.start

    @property
    def shape(self: _LinearMixinT) -> typ.Dict[str, int]:
        return dict(**super().shape, **self.broadcast_shapes(self._start_normalized, self._stop_normalized))


@dataclasses.dataclass(eq=False)
class LinearSpace(
    _SpaceMixin,
    _LinearMixin,
):
    @property
    def array(self: LinearSpaceT) -> kgpy.units.QuantityLike:
        shape = self.shape
        return np.linspace(
            start=self._start_normalized.array_aligned(shape)[..., 0],
            stop=self._stop_normalized.array_aligned(shape)[..., 0],
            num=self.num,
            axis=~0,
            endpoint=self.endpoint,
        )


@dataclasses.dataclass(eq=False)
class _RandomSpaceMixin(_SpaceMixin):

    seed: typ.Optional[int] = 42

    @property
    def _rng(self: _RandomSpaceMixinT) -> np.random.Generator:
        return np.random.default_rng(seed=self.seed)


@dataclasses.dataclass(eq=False)
class UniformRandomSpace(
    _RandomSpaceMixin,
    _LinearMixin[StartArrayT, StopArrayT],
):

    @property
    def array(self: UniformRandomSpaceT) -> kgpy.units.QuantityLike:
        shape = self.shape

        start = self.start_broadcasted.array_aligned(shape)
        stop = self.stop_broadcasted.array_aligned(shape)

        unit = None
        if isinstance(start, u.Quantity):
            unit = start.unit
            start = start.value
            stop = stop.to(unit).value

        value = self._rng.uniform(
            low=start,
            high=stop,
        )

        if unit is not None:
            value = value << unit

        return value


@dataclasses.dataclass(eq=False)
class _NormalMixin(
    AbstractArray[kgpy.units.QuantityLike],
    typ.Generic[CenterT, WidthT]
):

    center: CenterT = 0
    width: WidthT = 0

    @property
    def _center_normalized(self: _NormalMixinT) -> AbstractArray:
        if not isinstance(self.center, AbstractArray):
            return Array(self.center)
        else:
            return self.center

    @property
    def _width_normalized(self: _NormalMixinT) -> AbstractArray:
        if not isinstance(self.width, AbstractArray):
            return Array(self.width)
        else:
            return self.width

    @property
    def center_broadcasted(self: _NormalMixinT) -> Array:
        return np.broadcast_to(self._center_normalized, shape=self.shape, subok=True)

    @property
    def width_broadcasted(self: _NormalMixinT) -> Array:
        return np.broadcast_to(self._width_normalized, shape=self.shape, subok=True)

    @property
    def unit(self) -> typ.Optional[u.Unit]:
        unit = super().unit
        if hasattr(self.center, 'unit'):
            unit = self.center.unit
        return unit

    @property
    def shape(self: _NormalMixinT) -> typ.Dict[str, int]:
        return dict(**super().shape, **self.broadcast_shapes(self._width_normalized, self._center_normalized))


@dataclasses.dataclass(eq=False)
class NormalRandomSpace(
    _RandomSpaceMixin,
    _NormalMixin[CenterT, WidthT],
):

    @property
    def array(self: NormalRandomSpaceT) -> kgpy.units.QuantityLike:
        shape = self.shape

        center = self.center_broadcasted.array_aligned(shape)
        width = self.width_broadcasted.array_aligned(shape)

        unit = None
        if isinstance(center, u.Quantity):
            unit = center.unit
            center = center.value
            width = width.to(unit).value

        value = self._rng.normal(
            loc=center,
            scale=width,
        )

        if unit is not None:
            value = value << unit

        return value


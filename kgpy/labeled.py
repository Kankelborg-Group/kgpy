"""
N-dimensional arrays with labeled dimensions.

============
Introduction
============

:mod:`kgpy.labeled` is a module which exists to support the concept of n-dimensional array where each of the dimensions
are labeled by a string.
This concept is very similar to :class:`xarray.Variable`, except with support for :class:`astropy.units.Quantity`.
Also see the post `Tensors Considered Harmful <https://nlp.seas.harvard.edu/NamedTensor>`_.

Expressing an n-dimensional array in this way has several advantages.
First, the arrays will automatically broadcast against one another, without needing extra dimensions for alignment.

.. jupyter-execute::

    import kgpy.labeled

    x = kgpy.labeled.LinearSpace(0, 1, num=2, axis='x')
    y = kgpy.labeled.LinearSpace(0, 1, num=3, axis='y')
    z = x * y
    z

|br| Second, reduction-like operations such as :func:`numpy.sum()`, :func:`numpy.mean()`, :func:`numpy.min()`, etc. can
use the dimension label instead of the axis position.

.. jupyter-execute::

    z.sum(axis='x')

|br| Finally, elements or slices of the array can be accessed using the dimension label instead of inserting extra
slices or ellipses to select the appropriate axis.

.. jupyter-execute::

    z[dict(y=1)]

|br| Note that above we would love to be able to do ``z[y=1]``, however Python does not currently support keyword
arguments to the ``__get_item__`` dunder method.

===============
Creating Arrays
===============

The most important member of the :mod:`kgpy.labeled` module is the :class:`kgpy.labeled.Array` class.
This class is a composition of a :class:`numpy.ndarray` and a :class:`list` of strings labeling the dimensions.

Here is how you would explicitly create a :class:`kgpy.labeled.Array` from a :class:`numpy.ndarray` and a :class:`list`
of strings.

.. jupyter-execute::

    import numpy as np
    import kgpy.labeled

    kgpy.labeled.Array(np.linspace(0, 1, num=4), axes=['x'])

|br| Note that trying the above without specifying the ``axes`` argument results in an error since the number of axes
does not match the number of dimensions.

.. jupyter-execute::
    :raises:

    kgpy.labeled.Array(np.linspace(0, 1, num=4))

|br| However if the first argument is a scalar, the ``axes`` argument does not need to be specified

.. jupyter-execute::

    kgpy.labeled.Array(5)

|br| It is generally discouraged to create :class:`kgpy.labeled.Array` instances explicitly.
The above example can be accomplished by using the :class:`kgpy.labeled.LinearSpace` class.

.. jupyter-execute::

    kgpy.labeled.LinearSpace(0, 1, num=4, axis='x')

|br| In addition to :class:`kgpy.labeled.LinearSpace`, there is also :class:`kgpy.labeled.UniformRandomSpace` and
:class:`kgpy.labeled.NormalRandomSpace` to help with array creation.

.. |br| raw:: html

     <br>

"""

import typing as typ
import abc
import dataclasses
import random
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
_RangeMixinT = typ.TypeVar('_RangeMixinT', bound='_RangeMixin')
LinearSpaceT = typ.TypeVar('LinearSpaceT', bound='LinearSpace')
_RandomSpaceMixinT = typ.TypeVar('_RandomSpaceMixinT', bound='_RandomSpaceMixin')
UniformRandomSpaceT = typ.TypeVar('UniformRandomSpaceT', bound='UniformRandomSpace')
StratifiedRandomSpaceT = typ.TypeVar('StratifiedRandomSpaceT', bound='StratifiedRandomSpace')
CenterT = typ.TypeVar('CenterT', bound='ArrayLike')
WidthT = typ.TypeVar('WidthT', bound='ArrayLike')
_SymmetricMixinT = typ.TypeVar('_SymmetricMixinT', bound='SymmetricMixin')
NormalRandomSpaceT = typ.TypeVar('NormalRandomSpaceT', bound='NormalRandomSpace')


@dataclasses.dataclass(eq=False)
class NDArrayMethodsMixin:

    def broadcast_to(
            self: NDArrayMethodsMixinT,
            shape: typ.Dict[str, int],
    ) -> NDArrayMethodsMixinT:
        return np.broadcast_to(self, shape=shape)

    def reshape(
            self: NDArrayMethodsMixinT,
            shape: typ.Dict[str, int],
    ) -> NDArrayMethodsMixinT:
        return np.reshape(self, newshape=shape)

    def min(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
            initial: kgpy.units.QuantityLike = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.min(self, axis=axis, initial=initial, where=where)

    def max(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
            initial: kgpy.units.QuantityLike = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.max(self, axis=axis, initial=initial, where=where)

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

    def rms(
            self: NDArrayMethodsMixinT,
            axis: typ.Optional[typ.Union[str, typ.Sequence[str]]] = None,
            where: NDArrayMethodsMixinT = np._NoValue,
    ) -> NDArrayMethodsMixinT:
        return np.sqrt(np.mean(np.square(self), axis=axis, where=where))


@dataclasses.dataclass(eq=False)
class ArrayInterface(
    kgpy.mixin.Copyable,
    NDArrayMethodsMixin,
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
    typ.Generic[ArrT],
):

    @property
    @abc.abstractmethod
    def normalized(self: ArrayInterfaceT) -> ArrayInterfaceT:
        return self.copy_shallow()

    @property
    @abc.abstractmethod
    def array(self: ArrayInterfaceT) -> ArrT:
        pass

    @property
    @abc.abstractmethod
    def array_labeled(self: ArrayInterfaceT) -> ArrayInterfaceT:
        pass

    @property
    @abc.abstractmethod
    def shape(self: ArrayInterfaceT) -> typ.Dict[str, int]:
        pass

    @property
    def dtype(self: ArrayInterfaceT):
        return self.array.dtype

    @property
    @abc.abstractmethod
    def unit(self) -> typ.Union[float, u.Unit]:
        return 1

    @abc.abstractmethod
    def to(self: ArrayInterfaceT, unit: u.UnitBase) -> ArrayInterfaceT:
        pass

    @property
    def broadcasted(self: ArrayInterfaceT) -> ArrayInterfaceT:
        return np.broadcast_to(self, shape=self.shape)

    @property
    def centers(self: ArrayInterfaceT) -> ArrayInterfaceT:
        return self

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
            if axis in shape:
                shape.pop(axis)
        shape_tuple = tuple(shape.values())
        for index in np.ndindex(*shape_tuple):
            yield dict(zip(shape.keys(), index))

    @abc.abstractmethod
    def add_axes(self: ArrayInterfaceT, axes: typ.List) -> ArrayInterfaceT:
        pass

    @abc.abstractmethod
    def combine_axes(
            self: ArrayInterfaceT,
            axes: typ.Sequence[str],
            axis_new: typ.Optional[str] = None,
    ) -> ArrayInterfaceT:
        pass

    @abc.abstractmethod
    def aligned(self: ArrayInterfaceT, shape: typ.Dict[str, int]):
        pass


@dataclasses.dataclass(eq=False)
class AbstractArray(
    ArrayInterface[ArrT],
):

    type_array_primary: typ.ClassVar[typ.Type] = np.ndarray
    type_array_auxiliary: typ.ClassVar[typ.Tuple[typ.Type, ...]] = (bool, int, float, complex, np.generic)
    type_array: typ.ClassVar[typ.Tuple[typ.Type, ...]] = type_array_auxiliary + (type_array_primary, )

    @property
    def array_labeled(self: AbstractArrayT) -> AbstractArrayT:
        return self

    @property
    @abc.abstractmethod
    def axes(self: AbstractArrayT) -> typ.Optional[typ.List[str]]:
        return []

    @property
    @abc.abstractmethod
    def shape(self: AbstractArrayT) -> typ.Dict[str, int]:
        return dict()

    def to(self: AbstractArrayT, unit: u.Unit) -> ArrayT:
        array = self.array
        if not isinstance(array, u.Quantity):
            array = array << u.dimensionless_unscaled
        return Array(
            array=array.to(unit),
            axes=self.axes.copy(),
        )

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
        for axis_index, axis_name in enumerate(self.axes):
            source.append(axis_index)
            destination.append(list(shape.keys()).index(axis_name))
        value = np.moveaxis(a=value, source=source, destination=destination)
        return value

    def aligned(self: AbstractArrayT, shape: typ.Dict[str, int]) -> ArrayT:
        return Array(array=self.array_aligned(shape), axes=list(shape.keys()))

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

            axes_new = self.axes.copy()
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
            result = Array(array=self.array.copy(), axes=self.axes.copy())
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

            result = Array(array=self.array.copy(), axes=self.axes.copy())
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
                a=self.array,
                source=[index_axis_rows, index_axis_columns],
                destination=[~1, ~0],
            )

            axes_new = self.axes.copy()
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
                axes=self.axes.copy(),
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

        elif func is np.moveaxis:

            args = list(args)

            if args:
                a = args.pop(0)
            else:
                a = kwargs.pop('a')
            a = a.copy_shallow()

            if args:
                source = args.pop(0)
            else:
                source = kwargs.pop('source')

            if args:
                destination = args.pop(0)
            else:
                destination = kwargs.pop('destination')

            types_sequence = (list, tuple,)
            if not isinstance(source, types_sequence):
                source = (source, )
            if not isinstance(destination, types_sequence):
                destination = (destination, )

            for src, dest in zip(source, destination):
                a.axes[a.axes.index(src)] = dest

            return a

        elif func is np.reshape:
            args = list(args)
            if 'a' in kwargs:
                array = kwargs.pop('a')
            else:
                array = args.pop(0)

            if 'newshape' in kwargs:
                shape = kwargs.pop('newshape')
            else:
                shape = args.pop(0)

            return Array(
                array=np.reshape(array.array, tuple(shape.values())),
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
                    axes=self.axes.copy(),
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

        elif func is np.histogram2d:
            args = list(args)
            if args:
                x = args.pop(0)
            else:
                x = kwargs.pop('x')

            if args:
                y = args.pop(0)
            else:
                y = kwargs.pop('y')

            shape = kgpy.labeled.Array.broadcast_shapes(x, y)
            x = x.broadcast_to(shape)
            y = y.broadcast_to(shape)

            bins = kwargs.pop('bins')           # type: typ.Dict[str, int]
            if not isinstance(bins[next(iter(bins))], int):
                raise NotImplementedError
            range = kwargs.pop('range')
            weights = kwargs.pop('weights')

            key_x, key_y = bins.keys()

            shape_hist = shape.copy()
            shape_hist[key_x] = bins[key_x]
            shape_hist[key_y] = bins[key_y]

            shape_edges_x = shape_hist.copy()
            shape_edges_x[key_x] = shape_edges_x[key_x] + 1
            shape_edges_x.pop(key_y)

            shape_edges_y = shape_hist.copy()
            shape_edges_y[key_y] = shape_edges_y[key_y] + 1
            shape_edges_y.pop(key_x)

            hist = Array.empty(shape_hist)
            edges_x = Array.empty(shape_edges_x) * x.unit
            edges_y = Array.empty(shape_edges_y) * y.unit

            for index in x.ndindex(axis_ignored=(key_x, key_y)):
                if range is not None:
                    range_index = [[elem.array.value for elem in range[component]] for component in range]
                else:
                    range_index = None

                if weights is not None:
                    weights_index = weights[index].array.reshape(-1)
                else:
                    weights_index = None

                hist[index].array[:], edges_x[index].array[:], edges_y[index].array[:] = np.histogram2d(
                    x=x[index].array.reshape(-1),
                    y=y[index].array.reshape(-1),
                    bins=tuple(bins.values()),
                    range=range_index,
                    weights=weights_index,
                    **kwargs,
                )

            return hist, edges_x, edges_y

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
                            if kwargs['axis'] in axes_new:
                                axes_new.remove(kwargs['axis'])
                        else:
                            for axis in kwargs['axis']:
                                if axis in axes_new:
                                    axes_new.remove(axis)
                    else:
                        axes_new = []

            if 'axis' in kwargs:
                if kwargs['axis'] is None:
                    pass
                elif np.isscalar(kwargs['axis']):
                    if kwargs['axis'] in axes:
                        kwargs['axis'] = axes.index(kwargs['axis'])
                else:
                    kwargs['axis'] = tuple(axes.index(ax) for ax in kwargs['axis'] if ax in axes)

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
            raise ValueError(f'{func} not supported')

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
                source=[self.axes.index(axis) for axis in item.axes],
                destination=np.arange(len(item.axes)),
            )

            return Array(
                array=np.moveaxis(value[item.array], 0, ~0),
                axes=[axis for axis in self.axes if axis not in item.axes] + ['boolean']
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
                    axes_indices_advanced.append(self.axes.index(axis))
                    item_advanced[axis] = item_axis

            shape_advanced = self.broadcast_shapes(*item_advanced.values())

            value = np.moveaxis(
                a=self.array,
                source=axes_indices_advanced,
                destination=list(range(len(axes_indices_advanced))),
            )

            axes = self.axes.copy()
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
    def normalized(self: AbstractArrayT) -> ArrayT:
        other = super().normalized
        if isinstance(other.array, other.type_array_auxiliary):
            other.array = np.array(other.array)
        if other.axes is None:
            other.axes = []
        return other

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
            shape[self.axes[i]] = self.array.shape[i]
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
            axes = self.axes.copy()
            for axis in key_casted:
                item_axis = key_casted[axis]
                if isinstance(item_axis, int):
                    axes.remove(axis)
                if isinstance(item_axis, Array):
                    item_axis = item_axis.array_aligned(self.shape_broadcasted(item_axis))
                index[self.axes.index(axis)] = item_axis

            self.array[tuple(index)] = value.array_aligned({axis: 1 for axis in axes})


@dataclasses.dataclass(eq=False)
class Range(AbstractArray[np.ndarray]):

    start: int = 0
    stop: int = None
    step: int = 1
    axis: str = None

    @property
    def normalized(self: RangeT) -> RangeT:
        return super().normalized

    @property
    def unit(self) -> typ.Union[float, u.Unit]:
        unit = super().unit
        if hasattr(self.start, 'unit'):
            unit = self.start.unit
        return unit

    @property
    def shape(self: RangeT) -> typ.Dict[str, int]:
        shape = super().shape
        array = self.array
        for i in range(np.ndim(array)):
            shape[self.axes[i]] = array.shape[i]
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
        return super().axes + [self.axis]


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
class _RangeMixin(
    AbstractArray[kgpy.units.QuantityLike],
    typ.Generic[StartArrayT, StopArrayT],
):
    start: StartArrayT = None
    stop: StopArrayT = None

    @property
    def normalized(self: _RangeMixinT) -> _RangeMixinT:
        other = super().normalized
        if not isinstance(other.start, ArrayInterface):
            other.start = Array(other.start)
        if not isinstance(other.stop, ArrayInterface):
            other.stop = Array(other.stop)
        return other

    @property
    def unit(self) -> typ.Union[float, u.Unit]:
        unit = super().unit
        if hasattr(self.start, 'unit'):
            unit = self.start.unit
        return unit

    @property
    def range(self: _RangeMixinT) -> Array:
        return self.stop - self.start

    @property
    def shape(self: _RangeMixinT) -> typ.Dict[str, int]:
        norm = self.normalized
        return dict(**super().shape, **self.broadcast_shapes(norm.start, norm.stop))


@dataclasses.dataclass(eq=False)
class LinearSpace(
    _SpaceMixin,
    _RangeMixin[StartArrayT, StopArrayT],
):

    @property
    def array(self: LinearSpaceT) -> kgpy.units.QuantityLike:
        shape = self.shape
        shape.pop(self.axis)
        norm = self.normalized
        return np.linspace(
            start=norm.start.array_aligned(shape),
            stop=norm.stop.array_aligned(shape),
            num=self.num,
            axis=~0,
            endpoint=self.endpoint,
        )


@dataclasses.dataclass(eq=False)
class _RandomSpaceMixin(_SpaceMixin):

    seed: typ.Optional[int] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 10 ** 12)

    @property
    def _rng(self: _RandomSpaceMixinT) -> np.random.Generator:
        return np.random.default_rng(seed=self.seed)


@dataclasses.dataclass(eq=False)
class UniformRandomSpace(
    _RandomSpaceMixin,
    _RangeMixin[StartArrayT, StopArrayT],
):

    @property
    def array(self: UniformRandomSpaceT) -> kgpy.units.QuantityLike:
        shape = self.shape

        norm = self.normalized
        start = norm.start.broadcast_to(shape).array
        stop = norm.stop.broadcast_to(shape).array

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
class StratifiedRandomSpace(
    _RandomSpaceMixin,
    LinearSpace[StartArrayT, StopArrayT],
):
    shape_extra: typ.Dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def shape(self: StratifiedRandomSpaceT) -> typ.Dict[str, int]:
        return {**super().shape, **self.shape_extra}

    @property
    def array(self: StratifiedRandomSpaceT) -> kgpy.units.QuantityLike:
        result = super().array

        norm = self.normalized
        shape = norm.shape
        shape[norm.axis] = norm.num
        shape = {**shape, **self.shape_extra}
        step_size = norm.range / (norm.num - 1)
        step_size = step_size.broadcast_to(shape).array

        if isinstance(step_size, u.Quantity):
            unit = step_size.unit
            step_size = step_size.value
        else:
            unit = None

        delta = self._rng.uniform(
            low=-step_size / 2,
            high=step_size / 2,
        )

        if unit is not None:
            delta = delta << unit

        return result + delta

    @property
    def centers(self: StratifiedRandomSpaceT) -> LinearSpace:
        return LinearSpace(
            start=self.start,
            stop=self.stop,
            num=self.num,
            endpoint=self.endpoint,
            axis=self.axis,
        )


@dataclasses.dataclass(eq=False)
class _SymmetricMixin(
    AbstractArray[kgpy.units.QuantityLike],
    typ.Generic[CenterT, WidthT]
):

    center: CenterT = 0
    width: WidthT = 0

    @property
    def normalized(self: _SymmetricMixinT) -> _SymmetricMixinT:
        other = super().normalized
        if not isinstance(other.center, ArrayInterface):
            other.center = Array(other.center)
        if not isinstance(other.width, ArrayInterface):
            other.width = Array(other.width)
        return other

    @property
    def unit(self) -> typ.Optional[u.Unit]:
        unit = super().unit
        if hasattr(self.center, 'unit'):
            unit = self.center.unit
        return unit

    @property
    def shape(self: _SymmetricMixinT) -> typ.Dict[str, int]:
        norm = self.normalized
        return dict(**super().shape, **self.broadcast_shapes(norm.width, norm.center))


@dataclasses.dataclass(eq=False)
class NormalRandomSpace(
    _RandomSpaceMixin,
    _SymmetricMixin[CenterT, WidthT],
):

    @property
    def array(self: NormalRandomSpaceT) -> kgpy.units.QuantityLike:
        shape = self.shape

        norm = self.normalized
        center = norm.center.broadcast_to(shape).array
        width = norm.width.broadcast_to(shape).array

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


def indices(shape: typ.Dict[str, int]) -> Array:
    return kgpy.labeled.Array(
        array=np.indices(shape.values()),
        axes=['axis'] + list(shape.keys()),
    )

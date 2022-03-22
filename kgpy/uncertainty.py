import abc
import typing as typ
import dataclasses
import random
import numpy as np
import astropy.units as u
import kgpy.mixin
import kgpy.labeled

__all__ = [
    'AbstractArray',
    'ArrayLike',
    'Array',
    'Uniform',
    'Normal',
]

NominalT = typ.TypeVar('NominalT', bound=kgpy.labeled.ArrayLike)
DistributionT = typ.TypeVar('DistributionT', bound=kgpy.labeled.ArrayLike)
AbstractArrayT = typ.TypeVar('AbstractArrayT', bound='AbstractArray')
ArrayT = typ.TypeVar('ArrayT', bound='Array')
_DistributionT = typ.TypeVar('_DistributionT', bound='_Distribution')
_UniformBaseT = typ.TypeVar('_UniformBaseT', bound='_UniformBase')
UniformT = typ.TypeVar('UniformT', bound='Uniform')
NormalT = typ.TypeVar('NormalT', bound='Normal')


@dataclasses.dataclass(eq=False)
class AbstractArray(
    kgpy.labeled.ArrayInterface,
    typ.Generic[NominalT, DistributionT],
):
    type_array_primary: typ.ClassVar[typ.Type] = kgpy.labeled.AbstractArray
    type_array_auxiliary: typ.ClassVar[typ.Tuple[typ.Type, ...]] = kgpy.labeled.AbstractArray.type_array
    type_array: typ.ClassVar[typ.Tuple[typ.Type, ...]] = type_array_auxiliary + (type_array_primary, )

    nominal: NominalT = 0 * u.dimensionless_unscaled

    @property
    def normalized(self: AbstractArrayT) -> AbstractArrayT:
        other = super().normalized
        if isinstance(other.nominal, self.type_array_auxiliary):
            other.nominal = kgpy.labeled.Array(other.nominal)
        return other

    @property
    def array(self: AbstractArrayT) -> np.ndarray:
        return self.array_labeled.array

    @property
    def array_labeled(self: AbstractArrayT) -> AbstractArrayT:
        return np.concatenate(
            arrays=[self.nominal, self.distribution],
            axis=self.axis_distribution,
        )

    @property
    @abc.abstractmethod
    def distribution(self: AbstractArrayT) -> typ.Optional[DistributionT]:
        pass

    @property
    @abc.abstractmethod
    def axis_distribution(self: AbstractArrayT) -> str:
        return '_distribution'

    def to(self: AbstractArrayT, unit: u.Unit) -> ArrayT:
        return Array(
            nominal=self.nominal.to(unit),
            distribution=self.distribution.to(unit),
        )

    @property
    def unit(self: AbstractArrayT) -> typ.Union[float, u.Unit]:
        return getattr(self.nominal, 'unit', 1)

    @property
    def shape(self: AbstractArrayT) -> typ.Dict[str, int]:
        shape = kgpy.labeled.Array.broadcast_shapes(self.nominal, self.distribution)
        shape.pop(self.axis_distribution)
        return shape

    @property
    def shape_distribution(self: AbstractArrayT) -> typ.Dict[str, int]:
        if self.distribution is not None:
            return {self.axis_distribution: self.distribution.shape[self.axis_distribution]}
        else:
            return {}

    @property
    def shape_all(self: AbstractArrayT) -> typ.Dict[str, int]:
        return {**self.shape, **self.shape_distribution}

    def __array_ufunc__(
            self,
            function,
            method,
            *inputs,
            **kwargs,
    ) -> AbstractArrayT:

        inputs_normalized = []
        for inp in inputs:
            if isinstance(inp, self.type_array_auxiliary):
                inp = kgpy.labeled.Array(inp)
            elif isinstance(inp, self.type_array_primary):
                pass
            elif isinstance(inp, AbstractArray):
                pass
            else:
                return NotImplemented
            inputs_normalized.append(inp)
        inputs = inputs_normalized

        inputs_nominal = [inp.normalized.nominal if isinstance(inp, AbstractArray) else inp for inp in inputs]
        inputs_distribution = [inp.normalized.distribution if isinstance(inp, AbstractArray) else inp for inp in inputs]

        for inp_nominal, inp_distribution in zip(inputs_nominal, inputs_distribution):
            result_nominal = inp_nominal.__array_ufunc__(function, method, *inputs_nominal, **kwargs)
            if result_nominal is NotImplemented:
                continue
            result_distribution = inp_distribution.__array_ufunc__(function, method, *inputs_distribution, **kwargs)
            if result_distribution is NotImplemented:
                continue
            return Array(
                nominal=result_nominal,
                distribution=result_distribution,
            )

        return NotImplemented

    def __array_function__(
            self: AbstractArrayT,
            func: typ.Callable,
            types: typ.Collection,
            args: typ.Tuple,
            kwargs: typ.Dict[str, typ.Any],
    ) -> AbstractArrayT:

        if func is np.stack:
            args = list(args)
            if args:
                if 'arrays' in kwargs:
                    raise TypeError(f"{func} got multiple values for argument 'arrays'")
                arrays = args.pop(0)
            else:
                arrays = kwargs.pop('arrays')

            arrays_nominal = [a.nominal if isinstance(a, type(self)) else a for a in arrays]
            arrays_nominal = [a if isinstance(a, kgpy.labeled.AbstractArray) else kgpy.labeled.Array(a) for a in arrays_nominal]

            arrays_distribution = [a.distribution if isinstance(a, type(self)) else a for a in arrays]
            arrays_distribution = [a if isinstance(a, kgpy.labeled.AbstractArray) else kgpy.labeled.Array(a) for a in arrays_distribution]

            return Array(
                nominal=np.stack(arrays_nominal, *args, **kwargs),
                distribution=np.stack(arrays_distribution, *args, **kwargs),
            )

        elif func is np.broadcast_to:
            args = list(args)
            if args:
                array = args.pop(0)
            else:
                array = kwargs['array']

            if args:
                shape = args.pop(0)
            else:
                shape = kwargs['shape']

            shape_distribution = {**shape, **self.shape_distribution}

            array = array.normalized
            return Array(
                nominal=np.broadcast_to(array.nominal, shape),
                distribution=np.broadcast_to(array.distribution, shape_distribution),
            )

        elif func in [
            np.unravel_index,
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
            args_nominal = [arg.normalized.nominal if isinstance(arg, AbstractArray) else arg for arg in args]
            types_nominal = list(type(arg) for arg in args_nominal if getattr(arg, '__array_function__', None) is not None)
            args_distribution = [arg.distribution if isinstance(arg, AbstractArray) else arg for arg in args]
            types_distribution = list(type(arg) for arg in args_distribution if getattr(arg, '__array_function__', None) is not None)

            kwargs_nominal = {k: kwargs[k].normalized.nominal if isinstance(kwargs[k], AbstractArray) else kwargs[k] for k in kwargs}
            kwargs_distribution = {k: kwargs[k].normalized.distribution if isinstance(kwargs[k], AbstractArray) else kwargs[k] for k in kwargs}

            if args:
                arg_nominal = args_nominal[0]
                arg_distribution = args_distribution[0]
            else:
                arg_nominal = kwargs_nominal[next(iter(kwargs_nominal))]
                arg_distribution = kwargs_distribution[next(iter(kwargs_distribution))]

            return Array(
                nominal=arg_nominal.__array_function__(
                    func,
                    types_nominal,
                    args_nominal,
                    kwargs_nominal,
                ),
                distribution=arg_distribution.__array_function__(
                    func,
                    types_distribution,
                    args_distribution,
                    kwargs_distribution,
                )
            )
        else:
            raise NotImplementedError

    def __mul__(self: AbstractArrayT, other: typ.Union['ArrayLike', u.UnitBase]):
        if isinstance(other, u.UnitBase):
            return Array(
                nominal=self.nominal * other,
                distribution=self.distribution * other,
            )
        else:
            return super().__mul__(other)

    def __lshift__(self: AbstractArrayT, other: u.UnitBase) -> ArrayT:
        if isinstance(other, u.UnitBase):
            return Array(
                nominal=self.nominal << other,
                distribution=self.distribution << other,
            )
        else:
            return super().__lshift__(other)

    def __truediv__(self: AbstractArrayT, other: u.UnitBase) -> ArrayT:
        if isinstance(other, u.UnitBase):
            return Array(
                nominal=self.nominal / other,
                distribution=self.distribution / other,
            )
        else:
            return super().__truediv__(other)

    def __bool__(self: AbstractArrayT) -> bool:
        return self.nominal.__bool__() and self.distribution.__bool__()

    def __getitem__(
            self: AbstractArrayT,
            item: typ.Union[typ.Dict[str, typ.Union[int, slice, kgpy.labeled.AbstractArray, 'AbstractArray']], kgpy.labeled.AbstractArray, 'AbstractArray'],
    ) -> ArrayT:
        if isinstance(item, AbstractArray):
            return Array(
                nominal=self.nominal.__getitem__(item.nominal),
                distribution=self.distribution.__getitem__(item.distribution),
            )
        elif isinstance(item, dict):
            item_nominal = {key: item[key].nominal if isinstance(item[key], AbstractArray) else item[key] for key in item}
            item_distribution = {key: item[key].distribution if isinstance(item[key], AbstractArray) else item[key] for key in item}
            return Array(
                nominal=self.nominal.__getitem__(item_nominal),
                distribution=self.distribution.__getitem__(item_distribution),
            )
        else:
            if self.axis_distribution not in item.shape:
                shape = {**item.shape, **self.shape_distribution}
                item_distribution = np.broadcast_to(item, shape)
            else:
                item_distribution = item.shape
            return Array(
                nominal=self.nominal.__getitem__(item),
                distribution=self.distribution.__getitem__(item_distribution),
            )

    @property
    def num_samples(self) -> int:
        return self.shape[self.axis_distribution]

    def combine_axes(
            self: AbstractArrayT,
            axes: typ.Sequence[str],
            axis_new: typ.Optional[str] = None,
    ) -> AbstractArrayT:
        return Array(
            nominal=self.nominal.combine_axes(axes=axes, axis_new=axis_new),
            distribution=self.distribution.combine_axes(axes=axes, axis_new=axis_new),
        )

    def add_axes(
            self: AbstractArrayT,
            axes: typ.List[str],
    ) -> AbstractArrayT:
        return Array(
            nominal=self.nominal.add_axes(axes=axes),
            distribution=self.distribution.add_Axes(axes=axes),
        )

    def matrix_inverse(self, axis_rows: str, axis_columns: str):
        inverse_nominal = self.nominal.matrix_inverse(axis_rows=axis_rows, axis_columns=axis_columns)
        if self.distribution is not None:
            inverse_distribution = self.distribution.matrix_inverse(axis_rows=axis_rows, axis_columns=axis_columns)
        else:
            inverse_distribution = None
        return Array(
            nominal=inverse_nominal,
            distribution=inverse_distribution,
        )


ArrayLike = typ.Union[kgpy.labeled.ArrayLike, AbstractArray]


@dataclasses.dataclass(eq=False)
class _DistributionMixin(
    kgpy.mixin.Copyable,
):
    num_samples: int = 11
    axis_distribution: str = '_distribution'
    seed: typ.Optional[int] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 10 ** 12)

    @property
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=self.seed)


WidthT = typ.TypeVar('WidthT', bound=kgpy.labeled.ArrayLike)


@dataclasses.dataclass(eq=False)
class _UniformBase(
    AbstractArray[kgpy.labeled.Array, kgpy.labeled.Array],
    typ.Generic[WidthT],
):
    width: WidthT = 0 * u.dimensionless_unscaled

    @property
    def normalized(self: _UniformBaseT) -> _UniformBaseT:
        other = super().normalized
        if isinstance(other.width, self.type_array_auxiliary):
            other.width = kgpy.labeled.Array(other.width)
        return other


@dataclasses.dataclass(eq=False)
class Uniform(
    _DistributionMixin,
    _UniformBase[WidthT],
):

    @property
    def distribution(self: UniformT) -> kgpy.labeled.UniformRandomSpace:
        norm = self.normalized
        return kgpy.labeled.UniformRandomSpace(
            start=norm.nominal - norm.width,
            stop=norm.nominal + norm.width,
            num=norm.num_samples,
            axis=norm.axis_distribution,
            seed=norm.seed,
        )


@dataclasses.dataclass
class Normal(Uniform):

    @property
    def distribution(self: UniformT) -> kgpy.labeled.NormalRandomSpace:
        norm = self.normalized
        return kgpy.labeled.NormalRandomSpace(
            center=norm.nominal,
            width=norm.width,
            num=norm.num_samples,
            axis=norm.axis_distribution,
            seed=norm.seed,
        )


@dataclasses.dataclass(eq=False)
class Array(AbstractArray[NominalT, DistributionT]):

    distribution: typ.Optional[DistributionT] = None
    axis_distribution: str = '_distribution'

    @property
    def normalized(self: ArrayT) -> ArrayT:
        other = super().normalized
        if other.distribution is None:
            other.distribution = other.nominal
        if isinstance(other.distribution, self.type_array_auxiliary):
            other.distribution = kgpy.labeled.Array(other.distribution)
        return other

    def __setitem__(
            self: AbstractArrayT,
            key: typ.Union[typ.Dict[str, typ.Union[int, slice, kgpy.labeled.AbstractArray, 'AbstractArray']], kgpy.labeled.AbstractArray, 'AbstractArray'],
            value: typ.Union[bool, int, float, np.ndarray, kgpy.labeled.AbstractArray, 'AbstractArray'],
    ):
        if isinstance(value, AbstractArray):
            value_nominal = value.nominal
            value_distribution = value.distribution
        else:
            value_nominal = value_distribution = value

        if isinstance(key, AbstractArray):
            key_nominal = key.nominal
            key_distribution = key.distribution
        else:
            key_nominal = key_distribution = key
            if not isinstance(key_distribution, dict):
                key_distribution = np.broadcast_to(key_distribution, {**key_distribution.shape, **self.shape_distribution})

        self.nominal[key_nominal] = value_nominal
        if self.distribution is not None:
            self.distribution[key_distribution] = value_distribution

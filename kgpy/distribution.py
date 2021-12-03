import abc
import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.mixin
import kgpy.labeled

__all__ = [
    'AbstractArray',
    'Array',
    'Uniform',
    'Normal',
]

ValueT = typ.TypeVar('ValueT', bound=kgpy.labeled.ArrayLike)
DistributionT = typ.TypeVar('DistributionT', bound=kgpy.labeled.ArrayLike)
AbstractArrayT = typ.TypeVar('AbstractArrayT', bound='AbstractArray')
ArrayT = typ.TypeVar('ArrayT', bound='Array')
_DistributionT = typ.TypeVar('_DistributionT', bound='_Distribution')
_UniformBaseT = typ.TypeVar('_UniformBaseT', bound='_UniformBase')
UniformT = typ.TypeVar('UniformT', bound='Uniform')
NormalT = typ.TypeVar('NormalT', bound='Normal')


@dataclasses.dataclass(eq=False)
class AbstractArray(
    kgpy.mixin.Copyable,
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
    typ.Generic[ValueT, DistributionT],
):
    axis: typ.ClassVar[str] = '_distribution'

    value: ValueT = 0 * u.dimensionless_unscaled

    @classmethod
    def _normalize_parameter(
            cls: typ.Type[AbstractArrayT],
            parameter: kgpy.labeled.ArrayLike,
    ) -> kgpy.labeled.AbstractArray:

        if not isinstance(parameter, kgpy.labeled.AbstractArray):
            parameter = kgpy.labeled.Array(parameter)
        # if not isinstance(parameter.value, u.Quantity):
        #     parameter.value = parameter.value << u.dimensionless_unscaled
        return parameter

    @property
    def _value_normalized(self: AbstractArrayT) -> kgpy.labeled.AbstractArray[u.Quantity]:
        return self._normalize_parameter(self.value)

    @property
    @abc.abstractmethod
    def distribution(self: AbstractArrayT) -> DistributionT:
        pass

    @property
    def _distribution_normalized(self: AbstractArrayT) -> kgpy.labeled.AbstractArray:
        return self._normalize_parameter(self.distribution)

    @property
    def shape(self) -> typ.Dict[str, int]:
        return kgpy.labeled.Array.broadcast_shapes(self.value, self.distribution)

    def __array_ufunc__(
            self: AbstractArrayT,
            function: np.ufunc,
            method: str,
            *inputs: typ.Union[kgpy.labeled.ArrayLike, AbstractArrayT],
            **kwargs: typ.Any,
    ) -> AbstractArrayT:
        inputs = [Array(inp, inp) if not isinstance(inp, AbstractArray) else inp for inp in inputs]

        inputs_value = [inp.value for inp in inputs]
        inputs_distribution = [inp.distribution for inp in inputs]

        for inp in inputs:
            if hasattr(inp.value, '__array_ufunc__'):
                if hasattr(inp.distribution, '__array_ufunc__'):
                    result_value = inp.value.__array_ufunc__(function, method, *inputs_value, **kwargs)
                    if result_value is NotImplemented:
                        continue
                    result_distribution = inp.distribution.__array_ufunc__(function, method, *inputs_distribution, **kwargs)
                    if result_distribution is NotImplemented:
                        continue
                    return Array(
                        value=result_value,
                        distribution=result_distribution,
                    )

        raise ValueError

    def __array_function__(
            self: AbstractArrayT,
            function: typ.Callable,
            types: typ.Collection,
            args: typ.Tuple,
            kwargs: typ.Dict[str, typ.Any],
    ) -> AbstractArrayT:

        if function in [
            np.broadcast_to,
            np.unravel_index,
            np.stack,
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
            args_value = [arg.value if isinstance(arg, AbstractArray) else arg for arg in args]
            args_distribution = [arg.distribution if isinstance(arg, AbstractArray) else arg for arg in args]

            kwargs_value = {k: kwargs[k].value if isinstance(kwargs[k], AbstractArray) else kwargs[k] for k in kwargs}
            kwargs_distribution = {k: kwargs[k].distribution if isinstance(kwargs[k], AbstractArray) else kwargs[k] for k in kwargs}

            return Array(
                value=self.value.__array_function__(
                    function=function,
                    types=tuple(type(arg) for arg in args_value),
                    args=args_value,
                    kwargs=kwargs_value,
                ),
                distribution=self.distribution.__array_function__(
                    function=function,
                    types=tuple(type(arg) for arg in args_distribution),
                    args=args_distribution,
                    kwargs=kwargs_distribution,
                )
            )

    @property
    def num_samples(self) -> int:
        return self.shape[self.axis]

    def view(self: AbstractArrayT) -> AbstractArrayT:
        other = super().view()
        other.value = self.value
        return other

    def copy(self: AbstractArrayT) -> AbstractArrayT:
        other = super().copy()
        other.value = self.value.copy()
        return other


@dataclasses.dataclass(eq=False)
class _DistributionMixin(
    kgpy.mixin.Copyable,
):
    num_samples: int = 0
    seed: int = 42

    @property
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=self.seed)

    def view(self: _DistributionT) -> _DistributionT:
        other = super().view()
        other.num_samples = self.num_samples
        other.seed = self.seed
        return other

    def copy(self: _DistributionT) -> _DistributionT:
        other = super().copy()
        other.num_samples = self.num_samples
        other.seed = self.seed
        return other


WidthT = typ.TypeVar('WidthT', bound=kgpy.labeled.ArrayLike)


@dataclasses.dataclass(eq=False)
class _UniformBase(
    AbstractArray[kgpy.labeled.Array, kgpy.labeled.Array],
    typ.Generic[WidthT],
):
    width: WidthT = 0 * u.dimensionless_unscaled

    @property
    def _width_normalized(self: UniformT) -> kgpy.labeled.AbstractArray:
        return self._normalize_parameter(self.width)

    def view(self: UniformT) -> UniformT:
        other = super().view()
        other.width = self.width
        return other

    def copy(self: UniformT) -> UniformT:
        other = super().copy()
        other.width = self.width.copy()
        return other


@dataclasses.dataclass(eq=False)
class Uniform(
    _DistributionMixin,
    _UniformBase[WidthT],
):

    @property
    def distribution(self: UniformT) -> kgpy.labeled.UniformRandomSpace:
        return kgpy.labeled.UniformRandomSpace(
            start=self._value_normalized - self._width_normalized,
            stop=self._value_normalized + self._width_normalized,
            num=self.num_samples,
            axis=self.axis,
            seed=self.seed,
        )


@dataclasses.dataclass
class Normal(Uniform):

    @property
    def distribution(self: UniformT) -> kgpy.labeled.NormalRandomSpace:
        return kgpy.labeled.NormalRandomSpace(
            center=self._value_normalized,
            width=self._width_normalized,
            num=self.num_samples,
            axis=self.axis,
            seed=self.seed,

        )


@dataclasses.dataclass(eq=False)
class Array(AbstractArray[ValueT, DistributionT]):

    distribution: DistributionT = None

    def view(self: ArrayT) -> ArrayT:
        other = super().view()
        other.distribution = self.distribution
        return other

    def copy(self: ArrayT) -> ArrayT:
        other = super().copy()
        other.distribution = self.distribution.copy()
        return other

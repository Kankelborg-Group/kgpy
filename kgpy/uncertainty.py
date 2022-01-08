import abc
import typing as typ
import dataclasses
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
    kgpy.mixin.Copyable,
    kgpy.labeled.NDArrayMethodsMixin,
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
    typ.Generic[NominalT, DistributionT],
):
    type_array_primary: typ.ClassVar[typ.Type] = kgpy.labeled.AbstractArray
    type_array_auxiliary: typ.ClassVar[typ.Tuple[typ.Type, ...]] = kgpy.labeled.AbstractArray.type_array
    type_array: typ.ClassVar[typ.Tuple[typ.Type, ...]] = type_array_auxiliary + (type_array_primary, )

    axis_distribution: typ.ClassVar[str] = '_distribution'

    nominal: NominalT = 0 * u.dimensionless_unscaled

    @classmethod
    def _normalize_array(
            cls: typ.Type[AbstractArrayT],
            parameter: kgpy.labeled.ArrayLike,
    ) -> kgpy.labeled.AbstractArray:

        if not isinstance(parameter, kgpy.labeled.AbstractArray):
            parameter = kgpy.labeled.Array(parameter)
        # if not isinstance(parameter.value, u.Quantity):
        #     parameter.value = parameter.value << u.dimensionless_unscaled
        return parameter

    @property
    def nominal_normalized(self: AbstractArrayT) -> kgpy.labeled.AbstractArray[u.Quantity]:
        return self._normalize_array(self.nominal)

    @property
    @abc.abstractmethod
    def distribution(self: AbstractArrayT) -> typ.Optional[DistributionT]:
        pass

    @property
    def distribution_normalized(self: AbstractArrayT) -> kgpy.labeled.AbstractArray:
        if self.distribution is not None:
            return self._normalize_array(self.distribution)
        else:
            return self._nominal_normalized

    @property
    def unit(self: AbstractArrayT) -> typ.Optional[u.Unit]:
        if hasattr(self.nominal, 'unit'):
            return self.nominal.unit
        else:
            return None

    @property
    def shape(self) -> typ.Dict[str, int]:
        return kgpy.labeled.Array.broadcast_shapes(self.nominal, self.distribution)

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

        inputs_nominal = [inp.nominal_normalized if isinstance(inp, AbstractArray) else inp for inp in inputs]
        inputs_distribution = [inp.distribution_normalized if isinstance(inp, AbstractArray) else inp for inp in inputs]

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

        if func in [
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
            args_nominal = [arg.nominal_normalized if isinstance(arg, AbstractArray) else arg for arg in args]
            types_nominal = list(type(arg) for arg in args_nominal if getattr(arg, '__array_function__', None) is not None)
            args_distribution = [arg.distribution_normalized if isinstance(arg, AbstractArray) else arg for arg in args]
            types_distribution = list(type(arg) for arg in args_distribution if getattr(arg, '__array_function__', None) is not None)

            kwargs_nominal = {k: kwargs[k].nominal_normalized if isinstance(kwargs[k], AbstractArray) else kwargs[k] for k in kwargs}
            kwargs_distribution = {k: kwargs[k].distribution_normalized if isinstance(kwargs[k], AbstractArray) else kwargs[k] for k in kwargs}

            return Array(
                nominal=self.nominal_normalized.__array_function__(
                    func,
                    types_nominal,
                    args_nominal,
                    kwargs_nominal,
                ),
                distribution=self.distribution_normalized.__array_function__(
                    func,
                    types_distribution,
                    args_distribution,
                    kwargs_distribution,
                )
            )
        else:
            raise NotImplementedError

    def __bool__(self: AbstractArrayT) -> bool:
        return self.nominal.__bool__() and self.distribution.__bool__()

    @property
    def num_samples(self) -> int:
        return self.shape[self.axis_distribution]

    def view(self: AbstractArrayT) -> AbstractArrayT:
        other = super().view()
        other.nominal = self.nominal
        return other

    def copy(self: AbstractArrayT) -> AbstractArrayT:
        other = super().copy()
        other.nominal = self.nominal.copy()
        return other


ArrayLike = typ.Union[kgpy.labeled.ArrayLike, AbstractArray]


@dataclasses.dataclass(eq=False)
class _DistributionMixin(
    kgpy.mixin.Copyable,
):
    num_samples: int = 11
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
    def width_normalized(self: UniformT) -> kgpy.labeled.AbstractArray:
        return self._normalize_array(self.width)

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
            start=self.nominal_normalized - self.width_normalized,
            stop=self.nominal_normalized + self.width_normalized,
            num=self.num_samples,
            axis=self.axis_distribution,
            seed=self.seed,
        )


@dataclasses.dataclass
class Normal(Uniform):

    @property
    def distribution(self: UniformT) -> kgpy.labeled.NormalRandomSpace:
        return kgpy.labeled.NormalRandomSpace(
            center=self.nominal_normalized,
            width=self.width_normalized,
            num=self.num_samples,
            axis=self.axis_distribution,
            seed=self.seed,

        )


@dataclasses.dataclass(eq=False)
class Array(AbstractArray[NominalT, DistributionT]):

    distribution: typ.Optional[DistributionT] = None

    def view(self: ArrayT) -> ArrayT:
        other = super().view()
        other.distribution = self.distribution
        return other

    def copy(self: ArrayT) -> ArrayT:
        other = super().copy()
        other.distribution = self.distribution.copy()
        return other

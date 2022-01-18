"""
Package for easier manipulation of vectors than the usual numpy functions.
"""
import typing as typ
import abc
import dataclasses
import numpy as np
import numpy.typing
import astropy.units as u
import kgpy.units
import kgpy.labeled
import kgpy.uncertainty

__all__ = [
    'ix', 'iy', 'iz',
    'Vector',
    'Vector2D',
    'Vector3D',
    'xhat_factory', 'yhat_factory', 'zhat_factory',
    'x_hat', 'y_hat', 'z_hat',
]

ix = 0
iy = 1
iz = 2

XT = typ.TypeVar('XT', bound=kgpy.uncertainty.ArrayLike)
YT = typ.TypeVar('YT', bound=kgpy.uncertainty.ArrayLike)
ZT = typ.TypeVar('ZT', bound=kgpy.uncertainty.ArrayLike)
RadiusT = typ.TypeVar('RadiusT', bound=kgpy.uncertainty.ArrayLike)
AzimuthT = typ.TypeVar('AzimuthT', bound=kgpy.uncertainty.ArrayLike)
InclinationT = typ.TypeVar('InclinationT', bound=kgpy.uncertainty.ArrayLike)
AbstractVectorT = typ.TypeVar('AbstractVectorT', bound='AbstractVector')
Cartesian2DT = typ.TypeVar('Cartesian2DT', bound='Cartesian2D')
Cartesian3DT = typ.TypeVar('Cartesian3DT', bound='Cartesian3D')
PolarT = typ.TypeVar('PolarT', bound='Polar')
CylindricalT = typ.TypeVar('CylindricalT', bound='Cylindrical')
SphericalT = typ.TypeVar('SphericalT', bound='Spherical')

VectorLike = typ.Union[kgpy.uncertainty.ArrayLike, 'AbstractVector']
ItemArrayT = typ.Union[kgpy.labeled.AbstractArray, kgpy.uncertainty.AbstractArray, AbstractVectorT]


@dataclasses.dataclass(eq=False)
class AbstractVector(
    kgpy.mixin.Copyable,
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
):
    type_coordinates = kgpy.uncertainty.AbstractArray.type_array + (kgpy.uncertainty.AbstractArray, )

    @property
    def coordinates(self: AbstractVectorT) -> typ.Dict[str, VectorLike]:
        return {component: getattr(self, component) for component in self.components}

    @property
    def components(self: AbstractVectorT) -> typ.Tuple[str, ...]:
        return tuple(field.name for field in dataclasses.fields(self))

    @property
    def shape(self: AbstractVectorT) -> typ.Dict[str, int]:
        return kgpy.labeled.Array.broadcast_shapes(*self.coordinates.values())

    def __array_ufunc__(self, function, method, *inputs, **kwargs):

        components_result = dict()

        for component in self.components:
            inputs_component = []
            for inp in inputs:
                if type(inp) == type(self):
                    inp = getattr(inp, component)
                elif isinstance(inp, self.type_coordinates):
                    pass
                else:
                    return NotImplemented
                inputs_component.append(inp)

            for inp in inputs_component:
                if not hasattr(inp, '__array_ufunc__'):
                    inp = np.array(inp)
                result = inp.__array_ufunc__(function, method, *inputs_component, **kwargs)
                if result is not NotImplemented:
                    components_result[component] = result
                    break

            if component not in components_result:
                return NotImplemented

        return type(self)(**components_result)

    def __bool__(self: AbstractVectorT) -> bool:
        result = True
        coordinates = self.coordinates
        for component in coordinates:
            result = result and coordinates[component].__bool__()
        return result

    def __mul__(self: AbstractVectorT, other: typ.Union[VectorLike, u.Unit]) -> AbstractVectorT:
        if isinstance(other, u.Unit):
            coordinates = self.coordinates
            return type(self)(**{component: coordinates[component] * other for component in coordinates})
        else:
            return super().__mul__(other)

    def __lshift__(self: AbstractVectorT, other: typ.Union[VectorLike, u.Unit]) -> AbstractVectorT:
        if isinstance(other, u.Unit):
            coordinates = self.coordinates
            return type(self)(**{component: coordinates[component] << other for component in coordinates})
        else:
            return super().__lshift__(other)

    def __truediv__(self: AbstractVectorT, other: typ.Union[VectorLike, u.Unit]) -> AbstractVectorT:
        if isinstance(other, u.Unit):
            coordinates = self.coordinates
            return type(self)(**{component: coordinates[component] / other for component in coordinates})
        else:
            return super().__truediv__(other)

    def __array_function__(
            self: AbstractVectorT,
            func: typ.Callable,
            types: typ.Collection,
            args: typ.Tuple,
            kwargs: typ.Dict[str, typ.Any],
    ) -> AbstractVectorT:

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
            coordinates = dict()
            for component in self.components:
                args_component = [getattr(arg, component, arg) for arg in args]
                kwargs_component = {kw: getattr(kwargs[kw], component, kwargs[kw]) for kw in kwargs}
                coordinates[component] = func(*args_component, **kwargs_component)

            return type(self)(**coordinates)

        else:
            return NotImplemented

    def __getitem__(
            self: AbstractVectorT,
            item: typ.Union[typ.Dict[str, typ.Union[int, slice, ItemArrayT]], ItemArrayT],
    ):
        if isinstance(item, AbstractVector):
            coordinates = {c: getattr(self, c).__getitem__(getattr(item, c)) for c in self.components}
        elif isinstance(item, (kgpy.labeled.AbstractArray, kgpy.uncertainty.AbstractArray)):
            coordinates = {c: getattr(self, c).__getitem__(item) for c in self.components}
        elif isinstance(item, dict):
            coordinates = dict()
            for component in self.components:
                item_component = {k: getattr(item[k], component, item[k]) for k in item}
                coordinates[component] = getattr(self, component).__getitem__(item_component)
        else:
            raise TypeError
        return type(self)(**coordinates)

    def __setitem__(self, key, value):
        pass

    @property
    def length(self) -> kgpy.uncertainty.ArrayLike:
        result = 0
        coordinates = self.coordinates
        for component in coordinates:
            result = result + np.square(coordinates[component])
        result = np.sqrt(result)
        return result


@dataclasses.dataclass(eq=False)
class Cartesian2D(
    AbstractVector,
    typ.Generic[XT, YT],
):
    x: XT = 0
    y: YT = 0

    @classmethod
    def x_hat(cls: typ.Type[Cartesian2DT]) -> Cartesian2DT:
        return cls(x=1)

    @classmethod
    def y_hat(cls: typ.Type[Cartesian2DT]) -> Cartesian2DT:
        return cls(y=1)

    @property
    def length(self: Cartesian2DT) -> kgpy.uncertainty.ArrayLike:
        return np.sqrt(np.square(self.x) + np.square(self.y))

    @property
    def polar(self: Cartesian2DT) -> PolarT:
        return Polar(
            radius=np.sqrt(np.square(self.x) + np.square(self.y)),
            azimuth=np.arctan2(self.y, self.x)
        )

    def to_3d(self: Cartesian2DT, z: kgpy.uncertainty.ArrayLike) -> Cartesian3DT:
        return Cartesian3D(
            x=self.x,
            y=self.y,
            z=z,
        )


@dataclasses.dataclass(eq=False)
class Cartesian3D(
    Cartesian2D[XT, YT],
    typ.Generic[XT, YT, ZT],
):
    z: ZT = 0

    @property
    def length(self: Cartesian3DT) -> kgpy.uncertainty.ArrayLike:
        return np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))

    @property
    def xy(self: Cartesian3DT) -> Cartesian2DT:
        return Cartesian2D(
            x=self.x,
            y=self.y,
        )

    @property
    def cylindrical(self: Cartesian3DT) -> CylindricalT:
        return Cylindrical(
            radius=np.sqrt(np.square(self.x) + np.square(self.y)),
            azimuth=np.arctan2(self.y, self.x),
            z=self.z,
        )

    @property
    def spherical(self: Cartesian3DT) -> SphericalT:
        radius = np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))
        return Spherical(
            radius=radius,
            azimuth=np.arctan2(self.y, self.x),
            inclination=np.arccos(self.z / radius)
        )


@dataclasses.dataclass(eq=False)
class Polar(
    AbstractVector,
    typ.Generic[RadiusT, AzimuthT],
):
    radius: RadiusT = 0
    azimuth: AzimuthT = 0 * u.deg

    @property
    def length(self: PolarT) -> kgpy.uncertainty.ArrayLike:
        return self.radius

    @property
    def cartesian(self: PolarT) -> Cartesian2DT:
        return Cartesian2D(
            x=self.radius * np.cos(self.azimuth),
            y=self.radius * np.sin(self.azimuth),
        )


@dataclasses.dataclass(eq=False)
class Cylindrical(
    Polar[RadiusT, AzimuthT],
    typ.Generic[RadiusT, AzimuthT, ZT],
):
    z: ZT = 0

    @property
    def length(self: CylindricalT) -> kgpy.uncertainty.ArrayLike:
        return np.sqrt(np.square(self.radius) + np.square(self.z))

    @property
    def cartesian(self: CylindricalT) -> Cartesian3DT:
        return Cartesian3D(
            x=self.radius * np.cos(self.azimuth),
            y=self.radius * np.sin(self.azimuth),
            z=self.z,
        )
    
    @property
    def spherical(self: CylindricalT) -> SphericalT:
        return Spherical(
            radius=np.sqrt(np.square(self.radius) + np.square(self.z)),
            azimuth=self.azimuth,
            inclination=np.arctan(self.radius / self.z),
        )


@dataclasses.dataclass(eq=False)
class Spherical(
    Polar[RadiusT, AzimuthT],
    typ.Generic[RadiusT, AzimuthT, InclinationT],
):
    inclination: InclinationT = 0 * u.deg

    @property
    def cartesian(self: SphericalT) -> Cartesian3DT:
        return Cartesian3D(
            x=self.radius * np.cos(self.azimuth) * np.sin(self.inclination),
            y=self.radius * np.sin(self.azimuth) * np.sin(self.inclination),
            z=self.radius * np.cos(self.inclination),
        )

    @property
    def cylindrical(self: SphericalT) -> CylindricalT:
        return Cylindrical(
            radius=self.radius * np.sin(self.inclination),
            azimuth=self.azimuth,
            z=self.radius * np.cos(self.inclination),
        )


@dataclasses.dataclass(eq=False)
class Vector(
    kgpy.mixin.Copyable,
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
):

    @classmethod
    @abc.abstractmethod
    def dimensionless(cls) -> 'Vector':
        return cls()

    @classmethod
    @abc.abstractmethod
    def spatial(cls) -> 'Vector':
        return cls()

    @classmethod
    @abc.abstractmethod
    def angular(cls) -> 'Vector':
        return cls()

    @classmethod
    @abc.abstractmethod
    def from_quantity(cls, value: u.Quantity):
        return cls()

    @classmethod
    def from_tuple(cls, value: typ.Tuple):
        return cls()

    @property
    @abc.abstractmethod
    def quantity(self) -> u.Quantity:
        pass

    @abc.abstractmethod
    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        pass

    @abc.abstractmethod
    def __array_function__(self, function, types, args, kwargs):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    @abc.abstractmethod
    def __setitem__(self, key, value):
        pass

    @abc.abstractmethod
    def to_tuple(self):
        pass


@dataclasses.dataclass(eq=False)
class Vector2D(Vector):
    x: numpy.typing.ArrayLike = 0
    y: numpy.typing.ArrayLike = 0

    x_index: typ.ClassVar[int] = 0
    y_index: typ.ClassVar[int] = 1

    __array_priority__ = 100000

    @classmethod
    def dimensionless(cls) -> 'Vector2D':
        self = super().dimensionless()
        self.x = self.x * u.dimensionless_unscaled
        self.y = self.y * u.dimensionless_unscaled
        return self

    @classmethod
    def spatial(cls) -> 'Vector2D':
        self = super().spatial()
        self.x = self.x * u.mm
        self.y = self.y * u.mm
        return self

    @classmethod
    def angular(cls) -> 'Vector2D':
        self = super().angular()
        self.x = self.x * u.deg
        self.y = self.y * u.deg
        return self

    @classmethod
    def from_quantity(cls, value: u.Quantity):
        self = super().from_quantity(value)
        self.x = value[..., cls.x_index]
        self.y = value[..., cls.y_index]
        return self

    @classmethod
    def from_tuple(cls, value: typ.Tuple):
        self = super().from_tuple(value=value)
        self.x = value[ix]
        self.y = value[iy]
        return self

    @classmethod
    def from_cylindrical(
            cls,
            radius: u.Quantity = 0 * u.dimensionless_unscaled,
            azimuth: u.Quantity = 0 * u.deg,
    ) -> 'Vector2D':
        return cls(
            x=radius * np.cos(azimuth),
            y=radius * np.sin(azimuth),
        )

    @property
    def broadcast(self):
        return np.broadcast(self.x, self.y)

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.broadcast.shape

    @property
    def size(self) -> int:
        return self.broadcast.size

    @property
    def ndim(self) -> int:
        return self.broadcast.ndim

    @property
    def x_final(self) -> u.Quantity:
        return np.broadcast_to(self.x, self.shape, subok=True)

    @property
    def y_final(self) -> u.Quantity:
        return np.broadcast_to(self.y, self.shape, subok=True)

    def get_component(self, comp: str) -> u.Quantity:
        return getattr(self, comp)

    def set_component(self, comp: str, value: u.Quantity):
        setattr(self, comp, value)

    @property
    def quantity(self) -> u.Quantity:
        return np.stack([self.x_final, self.y_final], axis=~0)

    @property
    def length_squared(self):
        return np.square(self.x) + np.square(self.y)

    @property
    def length(self):
        return np.sqrt(self.length_squared)

    @property
    def length_l1(self):
        return self.x + self.y

    def normalize(self) -> 'Vector':
        return self / self.length

    @classmethod
    def _extract_attr(cls, values: typ.List, attr: str) -> typ.List:
        values_new = []
        for v in values:
            if isinstance(v, cls):
                values_new.append(getattr(v, attr))
            elif isinstance(v, list):
                values_new.append(cls._extract_attr(v, attr))
            else:
                values_new.append(v)
        return values_new

    @classmethod
    def _extract_attr_dict(cls, values: typ.Dict, attr: str) -> typ.Dict:
        values_new = dict()
        for key in values:
            v = values[key]
            if isinstance(v, cls):
                values_new[key] = getattr(v, attr)
            elif isinstance(v, list):
                values_new[key] = cls._extract_attr(v, attr)
            else:
                values_new[key] = v

        return values_new

    @classmethod
    def _extract_x(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'x')

    @classmethod
    def _extract_y(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'y')

    @classmethod
    def _extract_x_final(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'x_final')

    @classmethod
    def _extract_y_final(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'y_final')

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        inputs_x = self._extract_x(inputs)
        for in_x in inputs_x:
            if hasattr(in_x, '__array_ufunc__'):
                result_x = in_x.__array_ufunc__(function, method, *inputs_x, **kwargs)
                if result_x is not NotImplemented:
                    break
        inputs_y = self._extract_y(inputs)
        for in_y in inputs_y:
            if hasattr(in_y, '__array_ufunc__'):
                result_y = in_y.__array_ufunc__(function, method, *inputs_y, **kwargs)
                if result_y is not NotImplemented:
                    break
        if function is np.isfinite:
            return result_x & result_y
        elif function is np.equal:
            return result_x & result_y
        else:
            return type(self)(
                x=result_x,
                y=result_y,
            )

    def __array_function__(self, function, types, args, kwargs):
        if function is np.broadcast_to:
            return self._broadcast_to(*args, **kwargs)
        elif function is np.broadcast_arrays:
            return self._broadcast_arrays(*args, **kwargs)
        elif function is np.result_type:
            return type(self)
        elif function is np.ndim:
            return self.ndim
        elif function in [
            np.min, np.max, np.median, np.mean, np.sum, np.prod,
            np.stack,
            np.moveaxis, np.roll, np.nanmin, np.nanmax,
            np.nansum, np.nanmean, np.linspace, np.where, np.concatenate, np.take
        ]:
            return self._array_function_default(function, types, args, kwargs)
        else:
            raise NotImplementedError

        # args_x = tuple(self._extract_x_final(args))
        # args_y = tuple(self._extract_y_final(args))
        # types_x = [type(a) for a in args_x if getattr(a, '__array_function__', None) is not None]
        # types_y = [type(a) for a in args_y if getattr(a, '__array_function__', None) is not None]
        # result_x = self.x_final.__array_function__(function, types_x, args_x, kwargs)
        # result_y = self.y_final.__array_function__(function, types_y, args_y, kwargs)
        #
        # if isinstance(result_x, list):
        #     result = [type(self)(x=rx, y=ry) for rx, ry in zip(result_x, result_y)]
        # else:
        #     result = type(self)(x=result_x, y=result_y)
        # return result

    def _array_function_default(self, function, types, args, kwargs):
        args_x = tuple(self._extract_x_final(args))
        args_y = tuple(self._extract_y_final(args))
        kwargs_x = self._extract_attr_dict(kwargs, 'x')
        kwargs_y = self._extract_attr_dict(kwargs, 'y')
        types_x = [type(a) for a in args_x if getattr(a, '__array_function__', None) is not None]
        types_y = [type(a) for a in args_y if getattr(a, '__array_function__', None) is not None]
        return type(self)(
            x=self.x.__array_function__(function, types_x, args_x, kwargs_x),
            y=self.y.__array_function__(function, types_y, args_y, kwargs_y),
        )

    @classmethod
    def _broadcast_to(cls, value: 'Vector2D', shape: typ.Sequence[int], subok: bool = False) -> 'Vector2D':
        return cls(
            x=np.broadcast_to(value.x, shape, subok=subok),
            y=np.broadcast_to(value.y, shape, subok=subok),
        )

    @classmethod
    def _broadcast_arrays(cls, *args, **kwargs) -> typ.Iterator[numpy.typing.ArrayLike]:
        sh = np.broadcast_shapes(*[a.shape for a in args])
        for a in args:
            yield np.broadcast_to(a, sh, **kwargs)

    @classmethod
    def _min(cls, value: 'Vector2D'):
        return cls(
            x=value.x.min(),

        )

    # def __mul__(self, other) -> 'Vector2D':
    #     return type(self)(
    #         x=self.x.__mul__(other),
    #         y=self.y.__mul__(other),
    #     )
    #
    # def __truediv__(self, other) -> 'Vector2D':
    #     return type(self)(
    #         x=self.x.__truediv__(other),
    #         y=self.y.__truediv__(other),
    #     )

    # def __lshift__(self, other) -> 'Vector2D':
    #     return type(self)(
    #         x=self.x.__lshift__(other),
    #         y=self.y.__lshift__(other),
    #     )

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            return self.x * other.x + self.y * other.y
        else:
            return NotImplementedError

    def __getitem__(self, item):
        return type(self)(
            x=self.x_final.__getitem__(item),
            y=self.y_final.__getitem__(item),
        )

    def __setitem__(self, key, value):
        if isinstance(value, type(self)):
            self.x.__setitem__(key, value.x)
            self.y.__setitem__(key, value.y)
        else:
            self.x.__setitem__(key, value)
            self.y.__setitem__(key, value)

    # def __len__(self):
    #     return self.shape[0]

    def sum(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector2D':
        return np.sum(self, axis=axis, keepdims=keepdims)

    def min(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector2D':
        return np.min(self, axis=axis, keepdims=keepdims)

    def max(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector2D':
        return np.max(self, axis=axis, keepdims=keepdims)

    def reshape(self, *args) -> 'Vector2D':
        return type(self)(
            x=self.x_final.reshape(*args),
            y=self.y_final.reshape(*args),
        )

    def take(self, indices: numpy.typing.ArrayLike, axis: int = None, out: np.ndarray = None, mode: str = 'raise'):
        return np.take(a=self, indices=indices, axis=axis, out=out, mode=mode)

    def outer(self, other: 'Vector2D') -> 'matrix.Matrix2D':
        result = matrix.Matrix2D()
        result.xx = self.x * other.x
        result.xy = self.x * other.y
        result.yx = self.y * other.x
        result.yy = self.y * other.y
        return result

    def to(self, unit: u.Unit) -> 'Vector2D':
        return type(self)(
            x=self.x.to(unit),
            y=self.y.to(unit),
        )

    def to_3d(self, z: typ.Optional[u.Quantity] = None) -> 'Vector3D':
        other = Vector3D()
        other.x = self.x
        other.y = self.y
        if z is None:
            z = 0 * self.x
        other.z = z
        return other

    def to_tuple(self) -> typ.Tuple:
        return self.x, self.y


@dataclasses.dataclass(eq=False)
class Vector3D(Vector2D):
    z: numpy.typing.ArrayLike = 0
    z_index: typ.ClassVar[int] = 2

    __array_priority__ = 1000000

    @classmethod
    def from_quantity(cls, value: u.Quantity):
        self = super().from_quantity(value=value)
        self.z = value[..., cls.z_index]
        return self

    @classmethod
    def from_tuple(cls, value: typ.Tuple):
        self = super().from_tuple(value=value)
        self.z = value[iz]
        return self

    @classmethod
    def dimensionless(cls) -> 'Vector3D':
        self = super().dimensionless()    # type: Vector3D
        self.z = self.z * u.dimensionless_unscaled
        return self

    @classmethod
    def spatial(cls) -> 'Vector3D':
        self = super().spatial()    # type: Vector3D
        self.z = self.z * u.mm
        return self

    @classmethod
    def angular(cls) -> 'Vector3D':
        self = super().angular()    # type: Vector3D
        self.z = self.z * u.deg
        return self

    @classmethod
    def from_cylindrical(
            cls,
            radius: u.Quantity = 0 * u.dimensionless_unscaled,
            azimuth: u.Quantity = 0 * u.deg,
            z: u.Quantity = 0 * u.dimensionless_unscaled
    ) -> 'Vector3D':
        self = super().from_cylindrical(
            radius=radius,
            azimuth=azimuth,
        )  # type: Vector3D
        self.z = z
        return self

    @property
    def xy(self) -> Vector2D:
        return Vector2D(
            x=self.x,
            y=self.y,
        )

    @xy.setter
    def xy(self, value: Vector2D):
        self.x = value.x
        self.y = value.y

    @property
    def yz(self) -> Vector2D:
        return Vector2D(
            x=self.y,
            y=self.z,
        )

    @yz.setter
    def yz(self, value: Vector2D):
        self.y = value.x
        self.z = value.y

    @property
    def zx(self) -> Vector2D:
        return Vector2D(
            x=self.z,
            y=self.x,
        )

    @zx.setter
    def zx(self, value: Vector2D):
        self.z = value.x
        self.x = value.y

    @property
    def broadcast(self):
        return np.broadcast(super().broadcast, self.z)

    @property
    def z_final(self) -> u.Quantity:
        return np.broadcast_to(self.z, self.shape, subok=True)

    @property
    def quantity(self) -> u.Quantity:
        return np.stack([self.x_final, self.y_final, self.z_final], axis=~0)

    @property
    def length_squared(self) -> u.Quantity:
        return super().length_squared + np.square(self.z)

    @property
    def length_l1(self) -> u.Quantity:
        return super().length_l1 + self.z

    @classmethod
    def _extract_z(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'z')

    @classmethod
    def _extract_z_final(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'z_final')

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        result = super().__array_ufunc__(function, method, *inputs, **kwargs)
        inputs_z = self._extract_z(inputs)
        for in_z in inputs_z:
            if hasattr(in_z, '__array_ufunc__'):
                result_z = in_z.__array_ufunc__(function, method, *inputs_z, **kwargs)
                if result_z is not NotImplemented:
                    break
        if function is np.isfinite:
            return result & result_z
        elif function is np.equal:
            return result & result_z
        else:
            result.z = result_z
            return result

    # def __array_function__(self, function, types, args, kwargs):
    #     result = super().__array_function__(function, types, args, kwargs)
    #     args_z = tuple(self._extract_z_final(args))
    #     types_z = [type(a) for a in args_z if getattr(a, '__array_function__', None) is not None]
    #     result_z = self.z_final.__array_function__(function, types_z, args_z, kwargs)
    #
    #     if isinstance(result, list):
    #         for r, r_z in zip(result, result_z):
    #             r.z = r_z
    #     else:
    #         result.z = result_z
    #     return result

    def _array_function_default(self, function, types, args, kwargs):
        result = super()._array_function_default(function, types, args, kwargs)
        args_z = tuple(self._extract_z_final(args))
        kwargs_z = self._extract_attr_dict(kwargs, 'z')
        types_z = [type(a) for a in args_z if getattr(a, '__array_function__', None) is not None]
        result.z = self.z.__array_function__(function, types_z, args_z, kwargs_z)
        return result

        # args_x = tuple(self._extract_x_final(args))
        # args_y = tuple(self._extract_y_final(args))
        # types_x = [type(a) for a in args_x if getattr(a, '__array_function__', None) is not None]
        # types_y = [type(a) for a in args_y if getattr(a, '__array_function__', None) is not None]
        # return type(self)(
        #     x=self.x.__array_function__(function, types_x, args_x, kwargs),
        #     y=self.y.__array_function__(function, types_y, args_y, kwargs),
        # )

    @classmethod
    def _broadcast_to(cls, value: 'Vector3D', shape: typ.Sequence[int], subok: bool = False) -> 'Vector2D':
        result = super()._broadcast_to(value, shape, subok=subok)
        result.z = np.broadcast_to(value.z, shape, subok=subok)
        return result

    # def __mul__(self, other) -> 'Vector3D':
    #     result = super().__mul__(other)     # type: Vector3D
    #     result.z = self.z.__mul__(other)
    #     return result
    #
    # def __truediv__(self, other) -> 'Vector3D':
    #     result = super().__truediv__(other)  # type: Vector3D
    #     result.z = self.z.__truediv__(other)
    #     return result

    # def __lshift__(self, other) -> 'Vector3D':
    #     result = super().__lshift__(other)  # type: Vector3D
    #     result.z = self.z.__lshift__(other)
    #     return result

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            return super().__matmul__(other) + self.z * other.z
        else:
            return NotImplementedError

    def cross(self, other):
        if isinstance(other, type(self)):
            return type(self)(
                x=self.y * other.z - self.z * other.y,
                y=self.z * other.x - self.x * other.z,
                z=self.x * other.y - self.y * other.x,
            )
        else:
            return NotImplementedError

    def __getitem__(self, item):
        other = super().__getitem__(item)
        other.z = self.z_final.__getitem__(item)
        return other

    def __setitem__(self, key, value: 'Vector3D'):
        super().__setitem__(key, value)
        if isinstance(value, type(self)):
            self.z.__setitem__(key, value.z)
        else:
            self.z.__setitem__(key, value)

    def min(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector3D':
        return super().min(axis=axis, keepdims=keepdims)

    def max(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector3D':
        return super().max(axis=axis, keepdims=keepdims)

    def sum(
            self,
            axis: typ.Union[None, int, typ.Iterable[int], typ.Tuple[int]] = None,
            keepdims: bool = False,
    ) -> 'Vector3D':
        return super().sum(axis=axis, keepdims=keepdims)

    def reshape(self, *args) -> 'Vector3D':
        result = super().reshape(*args)
        result.z = self.z_final.reshape(*args)
        return result

    def outer(self, other: 'Vector3D') -> 'matrix.Matrix3D':
        result = super().outer(other).to_3d()
        result.xz = self.x * other.z
        result.yz = self.y * other.z
        result.zx = self.z * other.x
        result.zy = self.z * other.y
        result.zz = self.z * other.z
        return result

    def to(self, unit: u.Unit) -> 'Vector3D':
        other = super().to(unit)
        other.z = self.z.to(unit)
        return other

    def to_tuple(self) -> typ.Tuple:
        return super().to_tuple() + self.z


def xhat_factory():
    a = Vector3D.dimensionless()
    a.x = 1 * u.dimensionless_unscaled
    return a


def yhat_factory():
    a = Vector3D.dimensionless()
    a.y = 1 * u.dimensionless_unscaled
    return a


def zhat_factory():
    a = Vector3D.dimensionless()
    a.z = 1 * u.dimensionless_unscaled
    return a


x_hat = xhat_factory()
y_hat = yhat_factory()
z_hat = zhat_factory()

from . import matrix

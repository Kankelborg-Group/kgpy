"""
Package for easier manipulation of vectors than the usual numpy functions.
"""
import typing as typ
import abc
import dataclasses
import numpy as np
import numpy.typing
import astropy.units as u
from kgpy import units
from . import matrix

__all__ = [
    # 'x', 'y', 'z',
    'ix', 'iy', 'iz',
    # 'xy',
    # 'x_hat', 'y_hat', 'z_hat',
    # 'dot', 'outer', 'matmul', 'lefmatmul', 'length', 'normalize', 'from_components', 'from_components_cylindrical'
    'Vector',
    'Vector2D',
    'Vector3D',
    'xhat_factory', 'yhat_factory', 'zhat_factory',
    'x_hat', 'y_hat', 'z_hat',
]

ix = 0
iy = 1
iz = 2


#
# x = ..., ix
# y = ..., iy
# z = ..., iz
#
# xy = ..., slice(None, iz)
#
# x_hat = np.array([1, 0, 0])
# y_hat = np.array([0, 1, 0])
# z_hat = np.array([0, 0, 1])
#
#
# def to_3d(a: u.Quantity) -> u.Quantity:
#     return from_components(a[x], a[y])
#
#
# def dot(a: u.Quantity, b: u.Quantity, keepdims: bool = True) -> u.Quantity:
#     return np.sum(a * b, axis=~0, keepdims=keepdims)
#
#
# def outer(a: u.Quantity, b: u.Quantity) -> u.Quantity:
#     a = np.expand_dims(a, ~1)
#     b = np.expand_dims(b, ~0)
#     return a * b
#
#
# def matmul(a: u.Quantity, b: u.Quantity) -> u.Quantity:
#     b = np.expand_dims(b, ~0)
#     return matrix.mul(a, b)[..., 0]
#
#
# def lefmatmul(a: u.Quantity, b: u.Quantity) -> u.Quantity:
#     a = np.expand_dims(a, ~0)
#     return matrix.mul(a, b)[..., 0]
#
#
# def length(a: u.Quantity, keepdims: bool = True) -> u.Quantity:
#     return np.sqrt(np.sum(np.square(a), axis=~0, keepdims=keepdims))
#
#
# def normalize(a: u.Quantity, keepdims: bool = True) -> u.Quantity:
#     return a / length(a, keepdims=keepdims)
#
#
# def from_components(x: u.Quantity = 0, y: u.Quantity = 0, z: u.Quantity = 0, use_z: bool = True) -> u.Quantity:
#     x, y, z = np.broadcast_arrays(x, y, z, subok=True)
#
#     if use_z:
#         return np.stack([x, y, z], axis=~0)
#     else:
#         return np.stack([x, y], axis=~0)
#
#
# def from_components_cylindrical(r: u.Quantity = 0, phi: u.Quantity = 0, z: u.Quantity = 0) -> u.Quantity:
#     return from_components(r * np.cos(phi), r * np.sin(phi), z)


class Vector(
    np.lib.mixins.NDArrayOperatorsMixin,
    abc.ABC,
):

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


@dataclasses.dataclass
class Vector2D(Vector):
    x: numpy.typing.ArrayLike = 0 * u.dimensionless_unscaled
    y: numpy.typing.ArrayLike = 0 * u.dimensionless_unscaled

    x_index: typ.ClassVar[int] = 0
    y_index: typ.ClassVar[int] = 1

    __array_priority__ = 100000

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
        result_x = self.x.__array_ufunc__(function, method, *self._extract_x(inputs), **kwargs)
        result_y = self.y.__array_ufunc__(function, method, *self._extract_y(inputs), **kwargs)
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
        elif function in [
            np.min, np.max, np.median, np.mean, np.sum, np.prod, np.stack, np.moveaxis, np.roll, np.nanmin, np.nanmax,
            np.nansum, np.nanmean, np.linspace, np.where
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
        types_x = [type(a) for a in args_x if getattr(a, '__array_function__', None) is not None]
        types_y = [type(a) for a in args_y if getattr(a, '__array_function__', None) is not None]
        return type(self)(
            x=self.x.__array_function__(function, types_x, args_x, kwargs),
            y=self.y.__array_function__(function, types_y, args_y, kwargs),
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

    def copy(self) -> 'Vector2D':
        return type(self)(
            x=self.x.copy(),
            y=self.y.copy(),
        )


@dataclasses.dataclass
class Vector3D(Vector2D):
    z: numpy.typing.ArrayLike = 0 * u.dimensionless_unscaled
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
        result_z = self.z.__array_ufunc__(function, method, *self._extract_z(inputs), **kwargs)
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
        types_z = [type(a) for a in args_z if getattr(a, '__array_function__', None) is not None]
        result.z = self.z.__array_function__(function, types_z, args_z, kwargs)
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

    def copy(self) -> 'Vector3D':
        other = super().copy()
        other.z = self.z.copy()
        return other


def xhat_factory():
    return Vector3D(x=1 * u.dimensionless_unscaled)


def yhat_factory():
    return Vector3D(y=1 * u.dimensionless_unscaled)


def zhat_factory():
    return Vector3D(z=1 * u.dimensionless_unscaled)


x_hat = xhat_factory()
y_hat = yhat_factory()
z_hat = zhat_factory()

from . import matrix

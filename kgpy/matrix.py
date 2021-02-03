"""
Complement to the :mod:`kgpy.vector` package for matrices.
"""
import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from . import vector

# __all__ = [
#     'xx', 'xy', 'xz',
#     'yx', 'yy', 'yz',
#     'zx', 'zy', 'zz',
#     'mul'
# ]
#
#
# def transpose(a: u.Quantity) -> u.Quantity:
#     return np.swapaxes(a, ~0, ~1)
#
#
# def mul(a: u.Quantity, b: u.Quantity) -> u.Quantity:
#     a = np.expand_dims(a, ~0)
#     b = np.expand_dims(b, ~2)
#     return np.sum(a * b, axis=~1)
#
#
# xx = [[1, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]] * u.dimensionless_unscaled
#
# xy = [[0, 1, 0],
#       [0, 0, 0],
#       [0, 0, 0]] * u.dimensionless_unscaled
#
# xz = [[0, 0, 1],
#       [0, 0, 0],
#       [0, 0, 0]] * u.dimensionless_unscaled
#
# yx = [[0, 0, 0],
#       [1, 0, 0],
#       [0, 0, 0]] * u.dimensionless_unscaled
#
# yy = [[0, 0, 0],
#       [0, 1, 0],
#       [0, 0, 0]] * u.dimensionless_unscaled
#
# yz = [[0, 0, 0],
#       [0, 0, 1],
#       [0, 0, 0]] * u.dimensionless_unscaled
#
# zx = [[0, 0, 0],
#       [0, 0, 0],
#       [1, 0, 0]] * u.dimensionless_unscaled
#
# zy = [[0, 0, 0],
#       [0, 0, 0],
#       [0, 1, 0]] * u.dimensionless_unscaled
#
# zz = [[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 1]] * u.dimensionless_unscaled


@dataclasses.dataclass
class Matrix2D(np.lib.mixins.NDArrayOperatorsMixin):
    xx: u.Quantity = 0 * u.dimensionless_unscaled
    xy: u.Quantity = 0 * u.dimensionless_unscaled
    yx: u.Quantity = 0 * u.dimensionless_unscaled
    yy: u.Quantity = 0 * u.dimensionless_unscaled

    @property
    def broadcast(self):
        return np.broadcast(self.xx, self.xy, self.yx, self.yy)

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.broadcast.shape

    @property
    def xx_final(self) -> u.Quantity:
        return np.broadcast_to(self.xx, self.shape, subok=True)

    @property
    def xy_final(self) -> u.Quantity:
        return np.broadcast_to(self.xy, self.shape, subok=True)

    @property
    def yx_final(self) -> u.Quantity:
        return np.broadcast_to(self.yx, self.shape, subok=True)

    @property
    def yy_final(self) -> u.Quantity:
        return np.broadcast_to(self.yy, self.shape, subok=True)

    @property
    def value(self) -> u.Quantity:
        return np.stack([
            np.stack([self.xx_final, self.xy_final], axis=~0),
            np.stack([self.yx_final, self.yy_final], axis=~0),
        ], axis=~1)

    @property
    def determinant(self) -> u.Quantity:
        return self.xx * self.yy - self.xy * self.yx

    @classmethod
    def _extract_attr(cls, values: typ.List, attr: str) -> typ.List:
        values_new = []
        for v in values:
            if isinstance(v, cls):
                values_new.append(getattr(v, attr))
            elif isinstance(v, list):
                values_new.append(cls._extract_attr(v))
            else:
                values_new.append(v)
        return values_new

    @classmethod
    def _extract_xx(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'xx')

    @classmethod
    def _extract_xy(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'xy')

    @classmethod
    def _extract_yx(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'yx')

    @classmethod
    def _extract_yy(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'yy')

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        return type(self)(
            xx=self.xx.__array_ufunc__(function, method, *self._extract_xx(inputs), **kwargs),
            xy=self.xy.__array_ufunc__(function, method, *self._extract_xy(inputs), **kwargs),
            yx=self.yx.__array_ufunc__(function, method, *self._extract_yx(inputs), **kwargs),
            yy=self.yy.__array_ufunc__(function, method, *self._extract_yy(inputs), **kwargs),
        )

    def __array_function__(self, function, types, args, kwargs):
        result_xx = self.xx_final.__array_function__(function, types, tuple(self._extract_xx(args)), kwargs)
        result_xy = self.xy_final.__array_function__(function, types, tuple(self._extract_xy(args)), kwargs)
        result_yx = self.yx_final.__array_function__(function, types, tuple(self._extract_yx(args)), kwargs)
        result_yy = self.yy_final.__array_function__(function, types, tuple(self._extract_yy(args)), kwargs)

        if isinstance(result_xx, list):
            result = []
            for r_xx, r_xy, r_yx, r_yy in zip(result_xx, result_xy, result_yx, result_yy):
                result.append(type(self)(
                    xx=r_xx,
                    xy=r_xy,
                    yx=r_yx,
                    yy=r_yy,
                ))
        else:
            result = type(self)(
                xx=result_xx,
                xy=result_xy,
                yx=result_yx,
                yy=result_yy,
            )
        return result

    def __invert__(self) -> 'Matrix2D':
        result = type(self)()
        result.xx = self.yy
        result.yy = self.xx
        result.xy = -self.xy
        result.yx = -self.yx
        result = result / self.determinant
        return result

    def __matmul__(self, other):
        if isinstance(other, vector.Vector2D):
            return vector.Vector2D(
                x=self.xx * other.x + self.xy * other.y,
                y=self.yx * other.x + self.yy * other.y,
            )
        elif isinstance(other, type(self)):
            return type(self)(
                xx=self.xx * other.xx + self.xy * other.yx, xy=self.xx * other.xy + self.xy * other.yy,
                yx=self.yx * other.xx + self.yy * other.yx, yy=self.yx * other.xy + self.yy * other.yy,
            )
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        if isinstance(other, vector.Vector2D):
            return vector.Vector2D(
                x=other.x * self.xx + other.y * self.yx,
                y=other.x * self.xy + other.y * self.yy,
            )
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        return type(self)(
            xx=self.xx_final.__getitem__(item),
            xy=self.xy_final.__getitem__(item),
            yx=self.yx_final.__getitem__(item),
            yy=self.yy_final.__getitem__(item),
        )

    def __setitem__(self, key, value: 'Matrix2D'):
        self.xx.__setitem__(key, value.xx)
        self.xy.__setitem__(key, value.xy)
        self.yx.__setitem__(key, value.yx)
        self.yy.__setitem__(key, value.yy)

    def to_3d(self) -> 'Matrix3D':
        other = Matrix3D()
        other.xx = self.xx
        other.xy = self.xy
        other.yx = self.yx
        other.yy = self.yy
        return other

    def copy(self) -> 'Matrix2D':
        return type(self)(
            xx=self.xx.copy(),
            xy=self.xy.copy(),
            yx=self.yx.copy(),
            yy=self.yy.copy(),
        )


@dataclasses.dataclass
class Matrix3D(Matrix2D):
    xz: u.Quantity = 0 * u.dimensionless_unscaled
    yz: u.Quantity = 0 * u.dimensionless_unscaled
    zx: u.Quantity = 0 * u.dimensionless_unscaled
    zy: u.Quantity = 0 * u.dimensionless_unscaled
    zz: u.Quantity = 0 * u.dimensionless_unscaled

    @property
    def broadcast(self):
        return np.broadcast(super().broadcast, self.xz, self.yz, self.zx, self.zy, self.zz)

    @property
    def xz_final(self) -> u.Quantity:
        return np.broadcast_to(self.xz, self.shape, subok=True)

    @property
    def yz_final(self) -> u.Quantity:
        return np.broadcast_to(self.yz, self.shape, subok=True)

    @property
    def zx_final(self) -> u.Quantity:
        return np.broadcast_to(self.zx, self.shape, subok=True)

    @property
    def zy_final(self) -> u.Quantity:
        return np.broadcast_to(self.zy, self.shape, subok=True)

    @property
    def zz_final(self) -> u.Quantity:
        return np.broadcast_to(self.zz, self.shape, subok=True)

    @property
    def value(self) -> u.Quantity:
        return np.stack([
            np.stack([self.xx_final, self.xy_final, self.xz_final], axis=~0),
            np.stack([self.yx_final, self.yy_final, self.yz_final], axis=~0),
            np.stack([self.zx_final, self.zy_final, self.zz_final], axis=~0),
        ], axis=~1)

    @property
    def determinant(self) -> u.Quantity:
        dx = self.xx * (self.yy * self.zz - self.yz * self.zy)
        dy = self.xy * (self.yz * self.zx - self.yx * self.zz)
        dz = self.xz * (self.yx * self.zy - self.yy * self.zx)
        return dx + dy + dz

    @classmethod
    def _extract_xz(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'xz')

    @classmethod
    def _extract_yz(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'yz')

    @classmethod
    def _extract_zx(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'zx')

    @classmethod
    def _extract_zy(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'zy')

    @classmethod
    def _extract_zz(cls, values: typ.List) -> typ.List:
        return cls._extract_attr(values, 'zz')

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        result = super().__array_ufunc__(function, method, *inputs, **kwargs)
        result.xz = self.xz.__array_ufunc__(function, method, *self._extract_xz(inputs), **kwargs)
        result.yz = self.yz.__array_ufunc__(function, method, *self._extract_yz(inputs), **kwargs)
        result.zx = self.zx.__array_ufunc__(function, method, *self._extract_zx(inputs), **kwargs)
        result.zy = self.zy.__array_ufunc__(function, method, *self._extract_zy(inputs), **kwargs)
        result.zz = self.zz.__array_ufunc__(function, method, *self._extract_zz(inputs), **kwargs)
        return result

    def __array_function__(self, function, types, args, kwargs):
        result = super().__array_function__(function, types, args, kwargs)
        result_xz = self.xz_final.__array_function__(function, types, tuple(self._extract_xz(args)), kwargs)
        result_yz = self.yz_final.__array_function__(function, types, tuple(self._extract_yz(args)), kwargs)
        result_zx = self.zx_final.__array_function__(function, types, tuple(self._extract_zx(args)), kwargs)
        result_zy = self.zy_final.__array_function__(function, types, tuple(self._extract_zy(args)), kwargs)
        result_zz = self.zz_final.__array_function__(function, types, tuple(self._extract_zz(args)), kwargs)

        if isinstance(result, list):
            for r, r_xz, r_yz, r_zx, r_zy, r_zz in zip(result, result_xz, result_yz, result_zx, result_zy, result_zz):
                r.xz = r_xz
                r.yz = r_yz
                r.zx = r_zx
                r.zy = r_zy
                r.zz = r_zz
        else:
            result.xz = result_xz
            result.yz = result_yz
            result.zx = result_zx
            result.zy = result_zy
            result.zz = result_zz
        return result

    def __invert__(self) -> 'Matrix3D':
        raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, vector.Vector3D):
            return vector.Vector3D(
                x=self.xx * other.x + self.xy * other.y + self.xz * other.z,
                y=self.yx * other.x + self.yy * other.y + self.yz * other.z,
                z=self.zx * other.x + self.zy * other.y + self.zz * other.z,
            )
        elif isinstance(other, type(self)):
            return type(self)(
                xx=self.xx * other.xx + self.xy * other.yx + self.xz * other.zx, xy=self.xx * other.xy + self.xy * other.yy + self.xz * other.zy, xz=self.xx * other.xz + self.xy * other.yz + self.xz * other.zz,
                yx=self.yx * other.xx + self.yy * other.yx + self.yz * other.zx, yy=self.yx * other.xy + self.yy * other.yy + self.yz * other.zy, yz=self.yx * other.xz + self.yy * other.yz + self.yz * other.zz,
                zx=self.zx * other.xx + self.zy * other.yx + self.zz * other.zx, zy=self.zx * other.xy + self.zy * other.yy + self.zz * other.zy, zz=self.zx * other.xz + self.zy * other.yz + self.zz * other.zz,
            )
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        if isinstance(other, vector.Vector3D):
            return vector.Vector3D(
                x=other.x * self.xx + other.y * self.yx + other.z * self.zx,
                y=other.x * self.xy + other.y * self.yy + other.z * self.zy,
                z=other.x * self.xz + other.y * self.yz + other.z * self.zz,
            )
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        other = super().__getitem__(item)   # type: Matrix3D
        other.xz = self.xz_final.__getitem__(item)
        other.yz = self.yz_final.__getitem__(item)
        other.zx = self.zx_final.__getitem__(item)
        other.zy = self.zy_final.__getitem__(item)
        other.zz = self.zz_final.__getitem__(item)
        return other

    def __setitem__(self, key, value: 'Matrix3D'):
        super().__setitem__(key, value)
        self.xz.__setitem__(key, value.xz)
        self.yz.__setitem__(key, value.yz)
        self.zx.__setitem__(key, value.zx)
        self.zy.__setitem__(key, value.zy)
        self.zz.__setitem__(key, value.zz)

    def copy(self) -> 'Matrix3D':
        other = super().copy()      # type: Matrix3D
        other.xz = self.xz.copy()
        other.yz = self.yz.copy()
        other.zx = self.zx.copy()
        other.zy = self.zy.copy()
        other.zz = self.zz.copy()
        return other

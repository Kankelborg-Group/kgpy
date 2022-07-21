"""
Complement to the :mod:`kgpy.vector` package for matrices.
"""
import abc
import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors

__all__ = [
    'AbstractMatrix',
    'Cartesian2D',
    'Cartesian3D',
]


AbstractMatrixT = typ.TypeVar('AbstractMatrixT', bound='AbstractMatrix')
OtherAbstractMatrixT = typ.TypeVar('OtherAbstractMatrixT', bound='AbstractMatrix')
Cartesian1DT = typ.TypeVar('Cartesian1DT', bound='Cartesian1D')
Cartesian2DT = typ.TypeVar('Cartesian2DT', bound='Cartesian2D')
Cartesian3DT = typ.TypeVar('Cartesian3DT', bound='Cartesian3D')
CartesianNDT = typ.TypeVar('CartesianNDT', bound='CartesianND')


@dataclasses.dataclass(eq=False)
class AbstractMatrix(
    kgpy.vectors.AbstractVector,
):

    @classmethod
    @abc.abstractmethod
    def identity(cls: typ.Type[AbstractMatrixT]) -> AbstractMatrixT:
        pass

    @property
    def transpose(self: AbstractMatrixT) -> AbstractMatrixT:
        row_prototype = next(iter(self.coordinates.values()))
        result = type(row_prototype)().to_matrix()
        for component_column in row_prototype.coordinates:
            result.coordinates[component_column] = type(self)().to_vector()

        for component_row in self.coordinates:
            for component_column in self.coordinates[component_row].coordinates:
                result.coordinates[component_column].coordinates[component_row] = self.coordinates[component_row].coordinates[component_column]

        return result.to_matrix()

    def __matmul__(self: AbstractMatrixT, other: OtherAbstractMatrixT) -> AbstractMatrixT:
        if isinstance(other, AbstractMatrix):
            result = type(self)()
            other = other.transpose
            for self_component_row in self.coordinates:
                result.coordinates[self_component_row] = type(other)().to_vector()
                for other_component_row in other.coordinates:
                    element = self.coordinates[self_component_row] @ other.coordinates[other_component_row]
                    result.coordinates[self_component_row].coordinates[other_component_row] = element

        elif isinstance(other, kgpy.vectors.AbstractVector):
            result = type(self)().to_vector()
            for self_component_row in self.coordinates:
                result.coordinates[self_component_row] = self.coordinates[self_component_row] @ other

        else:
            result = self * other

        return result

    def inverse_numpy(self: AbstractMatrixT) -> AbstractMatrixT:

        unit_matrix = self.copy()
        for component_row in self.components:
            for component_column in self.coordinates[component_row].components:
                unit = 1 * getattr(self.coordinates[component_row].coordinates[component_column], 'unit', 1)
                unit_matrix.coordinates[component_row].coordinates[component_column] = unit

        value_matrix = self / unit_matrix

        axis_columns = 'column'
        axis_rows = 'row'
        arrays = []
        for component_row in value_matrix.components:
            array = kgpy.uncertainty.stack(list(value_matrix.coordinates[component_row].coordinates.values()), axis=axis_columns)
            arrays.append(array)
        arrays = np.stack(arrays, axis=axis_rows)

        arrays_inverse = arrays.matrix_inverse(axis_rows=axis_rows, axis_columns=axis_columns)
        axis_rows_inverse = axis_columns
        axis_columns_inverse = axis_rows

        inverse_matrix = self.transpose
        for i, component_row in enumerate(inverse_matrix.components):
            coordinates_row = inverse_matrix.coordinates[component_row]
            for j, component_column in enumerate(coordinates_row.components):
                element = arrays_inverse[{axis_rows_inverse: i, axis_columns_inverse: j}]
                coordinates_row.coordinates[component_column] = element

        inverse_matrix = inverse_matrix / unit_matrix.transpose
        return inverse_matrix

    def inverse_schulz(self: AbstractMatrixT, max_iterations: int = 100):

        inverse = 2 * self.transpose / np.square(self.length)

        for i in range(max_iterations):
            inverse_new = 2 * inverse - inverse @ self @ inverse
            if np.all(np.abs((inverse_new - inverse) / inverse) < 1e-10):
                print('max_iteration', i)
                return inverse_new
            inverse = inverse_new

        raise ValueError

    def __invert__(self: AbstractMatrixT) -> AbstractMatrixT:
        return self.inverse_numpy()

    @abc.abstractmethod
    def to_vector(self: AbstractMatrixT) -> kgpy.vectors.AbstractVector:
        pass


@dataclasses.dataclass(eq=False)
class Cartesian1D(
    AbstractMatrix,
    kgpy.vectors.Cartesian1D[kgpy.vectors.Cartesian1D],
):
    @classmethod
    def identity(cls: typ.Type[Cartesian1DT]) -> Cartesian1DT:
        return cls(
            x=kgpy.vectors.Cartesian1D(x=1),
        )

    @property
    def transpose(self: Cartesian1DT) -> Cartesian1DT:
        return self

    def to_vector(self: Cartesian1DT) -> kgpy.vectors.Cartesian1D:
        return kgpy.vectors.Cartesian1D(self.x)


@dataclasses.dataclass(eq=False)
class Cartesian2D(
    AbstractMatrix,
    kgpy.vectors.Cartesian2D[kgpy.vectors.Cartesian2D, kgpy.vectors.Cartesian2D],
):

    @classmethod
    def identity(cls: typ.Type[Cartesian2DT]) -> Cartesian2DT:
        return cls(
            x=kgpy.vectors.Cartesian2D(x=1, y=0),
            y=kgpy.vectors.Cartesian2D(x=0, y=1),
        )

    @classmethod
    def rotation(cls: typ.Type[Cartesian2DT], angle: kgpy.uncertainty.ArrayLike) -> Cartesian2DT:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return cls(
            x=kgpy.vectors.Cartesian2D(x=cos_a, y=-sin_a),
            y=kgpy.vectors.Cartesian2D(x=sin_a, y=cos_a),
        )

    def to_vector(self: Cartesian2DT) -> kgpy.vectors.Cartesian2D:
        return kgpy.vectors.Cartesian2D(x=self.x, y=self.y)


@dataclasses.dataclass(eq=False)
class Cartesian3D(
    AbstractMatrix,
    kgpy.vectors.Cartesian3D[
        kgpy.vectors.Cartesian3D,
        kgpy.vectors.Cartesian3D,
        kgpy.vectors.Cartesian3D,
    ],
):

    @classmethod
    def identity(cls: typ.Type[Cartesian3DT]) -> Cartesian3DT:
        return cls(
            x=kgpy.vectors.Cartesian3D(x=1, y=0, z=0),
            y=kgpy.vectors.Cartesian3D(x=0, y=1, z=0),
            z=kgpy.vectors.Cartesian3D(x=0, y=0, z=1),
        )

    @classmethod
    def rotation_x(cls: typ.Type[Cartesian3DT], angle: kgpy.uncertainty.ArrayLike) -> Cartesian3DT:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return cls(
            x=kgpy.vectors.Cartesian3D(x=1, y=0, z=0),
            y=kgpy.vectors.Cartesian3D(x=0, y=cos_a, z=-sin_a),
            z=kgpy.vectors.Cartesian3D(x=0, y=sin_a, z=cos_a),
        )

    @classmethod
    def rotation_y(cls: typ.Type[Cartesian3DT], angle: kgpy.uncertainty.ArrayLike) -> Cartesian3DT:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return cls(
            x=kgpy.vectors.Cartesian3D(x=cos_a, y=0, z=sin_a),
            y=kgpy.vectors.Cartesian3D(x=0, y=1, z=0),
            z=kgpy.vectors.Cartesian3D(x=-sin_a, y=0, z=cos_a),
        )

    @classmethod
    def rotation_z(cls: typ.Type[Cartesian3DT], angle: kgpy.uncertainty.ArrayLike) -> Cartesian3DT:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return cls(
            x=kgpy.vectors.Cartesian3D(x=cos_a, y=-sin_a, z=0),
            y=kgpy.vectors.Cartesian3D(x=sin_a, y=cos_a, z=0),
            z=kgpy.vectors.Cartesian3D(x=0, y=0, z=1),
        )

    def to_vector(self: Cartesian3DT) -> kgpy.vectors.Cartesian3D:
        return kgpy.vectors.Cartesian3D(x=self.x, y=self.y, z=self.z)


@dataclasses.dataclass(eq=False)
class CartesianND(
    kgpy.vectors.CartesianND[kgpy.vectors.CartesianND],
    AbstractMatrix,
):
    @classmethod
    def identity(cls: typ.Type[AbstractMatrixT]) -> AbstractMatrixT:
        raise NotImplementedError

    @property
    def determinant(self: AbstractMatrixT) -> kgpy.uncertainty.ArrayLike:
        raise NotImplementedError

    def to_vector(self: CartesianNDT) -> kgpy.vectors.CartesianNDT:
        return kgpy.vectors.CartesianND(self.coordinates)


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

    @property
    def transpose(self) -> 'Matrix2D':
        return type(self)(
            xx=self.xx,
            xy=self.yx,
            yx=self.xy,
            yy=self.yy,
        )

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
        if isinstance(other, kgpy.vector.Vector2D):
            return kgpy.vector.Vector2D(
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
        if isinstance(other, kgpy.vector.Vector2D):
            return kgpy.vector.Vector2D(
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

    @property
    def transpose(self) -> 'Matrix3D':
        other = super().transpose
        other.xz = self.zx
        other.yz = self.zy
        other.zx = self.xz
        other.zy = self.yz
        other.zz = self.zz
        return other

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
        if isinstance(other, kgpy.vector.Vector3D):
            return kgpy.vector.Vector3D(
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
        if isinstance(other, kgpy.vector.Vector3D):
            return kgpy.vector.Vector3D(
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

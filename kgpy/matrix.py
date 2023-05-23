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

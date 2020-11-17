import pytest
import astropy.units as u
from . import matrix
from .matrix import xx, xy, yx, yy

magic1 = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]] * u.dimensionless_unscaled

magic2 = [[10, 11, 12],
          [13, 14, 15],
          [16, 17, 18]] * u.dimensionless_unscaled

magic3 = [[19, 20, 21],
          [22, 23, 24],
          [25, 26, 27]] * u.dimensionless_unscaled

test_matrices = [magic1, magic2, magic3]


@pytest.mark.parametrize(
    argnames='mat',
    argvalues=[[a] for a in test_matrices],
)
def test_transpose_involution(mat: u.Quantity):
    b = matrix.transpose(matrix.transpose(mat))
    assert (b == mat).all()


@pytest.mark.parametrize(
    argnames='mat1, mat2',
    argvalues=[[a, b] for a in test_matrices for b in test_matrices]
)
def test_transpose_addition(mat1: u.Quantity, mat2: u.Quantity):
    b = matrix.transpose(mat1 + mat2)
    c = matrix.transpose(mat1) + matrix.transpose(mat2)
    assert (b == c).all()


@pytest.mark.parametrize(
    argnames='mat1, mat2',
    argvalues=[[a, b] for a in test_matrices for b in test_matrices]
)
def test_transpose_multiplication(mat1: u.Quantity, mat2: u.Quantity):
    b = matrix.transpose(matrix.mul(mat1, mat2))
    c = matrix.mul(matrix.transpose(mat2), matrix.transpose(mat1))
    assert (b == c).all()


@pytest.mark.parametrize(
    argnames='mat1, mat2, mat_out',
    argvalues=[
        [xx, xx, xx],
        [xy, yx, xx],
        [yx, xy, yy],
        [yy, yy, yy],
    ],
)
def test_mul(mat1: u.Quantity, mat2: u.Quantity, mat_out: u.Quantity):
    b = matrix.mul(mat1, mat2)
    assert (b == mat_out).all()


@pytest.mark.parametrize(
    argnames='mat1, mat2, mat3',
    argvalues=[[a, b, c] for a in test_matrices for b in test_matrices for c in test_matrices]
)
def test_mul_distributivity(mat1: u.Quantity, mat2: u.Quantity, mat3: u.Quantity):
    b = matrix.mul(mat1, mat2 + mat3)
    c = matrix.mul(mat1, mat2) + matrix.mul(mat1, mat3)
    assert (b == c).all()


@pytest.mark.parametrize(
    argnames='mat1, mat2, mat3',
    argvalues=[[a, b, c] for a in test_matrices for b in test_matrices for c in test_matrices]
)
def test_mul_associativity(mat1: u.Quantity, mat2: u.Quantity, mat3: u.Quantity):
    b = matrix.mul(matrix.mul(mat1, mat2), mat3)
    c = matrix.mul(mat1, matrix.mul(mat2, mat3))
    assert (b == c).all()
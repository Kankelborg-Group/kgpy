import pytest
import astropy.units as u
from . import vector, matrix
from .vector import x_hat, y_hat, z_hat
from .matrix import xx, xy, xz, yx, yy, yz, zx, zy, zz

magic1 = [1, 2, 3] * u.dimensionless_unscaled
magic2 = [4, 5, 6] * u.dimensionless_unscaled
magic3 = [7, 8, 9] * u.dimensionless_unscaled

unit_vectors = [x_hat, y_hat, z_hat]
test_vectors = [magic1, magic2, magic3]
test_matrices = [[xx, xy, xz],
                 [yx, yy, yz],
                 [zx, zy, zz]]


@pytest.mark.parametrize(
    argnames='vec1, vec2, out',
    argvalues=[[a, b, float(i == j)] for i, a in enumerate(unit_vectors) for j, b in enumerate(unit_vectors)],
)
def test_dot(vec1: u.Quantity, vec2: u.Quantity, out: u.Quantity):
    assert vector.dot(vec1, vec2) == out


@pytest.mark.parametrize(
    argnames='vec1, vec2',
    argvalues=[[a, b] for a in test_vectors for b in test_vectors]
)
def test_dot_commutative(vec1: u.Quantity, vec2: u.Quantity):
    assert vector.dot(vec1, vec2) == vector.dot(vec2, vec1)


@pytest.mark.parametrize(
    argnames='vec1, vec2, vec3',
    argvalues=[[a, b, c] for a in test_vectors for b in test_vectors for c in test_vectors]
)
def test_dot_distributive(vec1: u.Quantity, vec2: u.Quantity, vec3: u.Quantity):
    assert vector.dot(vec1, vec2 + vec3) == vector.dot(vec1, vec2) + vector.dot(vec1, vec3)


@pytest.mark.parametrize(
    argnames='vec1, vec2, mat_out',
    argvalues=[[b, a, c] for a, row in zip(unit_vectors, test_matrices) for b, c in zip(unit_vectors, row)],
)
def test_outer(vec1: u.Quantity, vec2: u.Quantity, mat_out: u.Quantity):
    assert (vector.outer(vec1, vec2) == mat_out).all()


@pytest.mark.parametrize(
    argnames='vec1, vec2',
    argvalues=[[a, b] for a in test_vectors for b in test_vectors]
)
def test_outer_transpose(vec1: u.Quantity, vec2: u.Quantity):
    assert (matrix.transpose(vector.outer(vec1, vec2)) == vector.outer(vec2, vec1)).all()


@pytest.mark.parametrize(
    argnames='vec1, vec2, vec3',
    argvalues=[[a, b, c] for a in test_vectors for b in test_vectors for c in test_vectors],
)
def test_outer_distributivity(vec1: u.Quantity, vec2: u.Quantity, vec3: u.Quantity):
    assert (vector.outer(vec1 + vec2, vec3) == vector.outer(vec1, vec3) + vector.outer(vec2, vec3)).all()


@pytest.mark.parametrize(
    argnames='mat, vec, vec_out',
    argvalues=[[c, b, a] for a, row in zip(unit_vectors, test_matrices) for b, c in zip(unit_vectors, row)],
)
def test_matmul(mat: u.Quantity, vec: u.Quantity, vec_out: u.Quantity):
    assert (vector.matmul(mat, vec) == vec_out).all()

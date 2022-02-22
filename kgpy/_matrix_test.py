import pytest
import numpy as np
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vector
import kgpy.matrix


@pytest.mark.parametrize(
    argnames='xy',
    argvalues=[
        2,
        kgpy.labeled.LinearSpace(2, 3, 3, axis='x')
    ],
)
@pytest.mark.parametrize(
    argnames='yx',
    argvalues=[
        3,
        kgpy.labeled.LinearSpace(3, 4, 4, axis='y'),
    ],
)
@pytest.mark.parametrize(
    argnames='width_xy',
    argvalues=[
        None,
        1,
    ],
)
@pytest.mark.parametrize(
    argnames='width_yx',
    argvalues=[
        None,
        1,
    ],
)
@pytest.mark.parametrize(
    argnames='unit',
    argvalues=[
        1,
        u.mm,
    ],
)
class TestCartesian2D:

    def _calc_matrix(
            self,
            xy: kgpy.labeled.ArrayLike,
            yx: kgpy.labeled.ArrayLike,
            width_xy: kgpy.labeled.ArrayLike,
            width_yx: kgpy.labeled.ArrayLike,
            unit: u.Unit,
    ):
        if width_xy is not None:
            xy = kgpy.uncertainty.Uniform(xy, width=width_xy)
        if width_yx is not None:
            yx = kgpy.uncertainty.Uniform(yx, width=width_yx)
        a = kgpy.matrix.Cartesian2D(
            x=kgpy.vector.Cartesian2D(x=kgpy.labeled.Array(1), y=xy),
            y=kgpy.vector.Cartesian2D(x=yx, y=kgpy.labeled.Array(4)),
        )
        return a * unit

    def test_transpose(
            self,
            xy: kgpy.labeled.ArrayLike,
            yx: kgpy.labeled.ArrayLike,
            width_xy: kgpy.labeled.ArrayLike,
            width_yx: kgpy.labeled.ArrayLike,
            unit: u.Unit,
    ):
        a = self._calc_matrix(xy, yx, width_xy, width_yx, unit)
        assert not np.all(a == a.transpose)
        assert np.all(a == a.transpose.transpose)

    def test__matmul__(
            self,
            xy: kgpy.labeled.ArrayLike,
            yx: kgpy.labeled.ArrayLike,
            width_xy: kgpy.labeled.ArrayLike,
            width_yx: kgpy.labeled.ArrayLike,
            unit: u.Unit,
    ):
        a = self._calc_matrix(xy, yx, width_xy, width_yx, unit)
        assert np.all(a @ (a + a) == a @ a + a @ a)
        assert np.all((a + a) @ a == a @ a + a @ a)
        assert np.all((a @ a).transpose == a.transpose @ a.transpose)

    def test__inverse_numpy(
            self,
            xy: kgpy.labeled.ArrayLike,
            yx: kgpy.labeled.ArrayLike,
            width_xy: kgpy.labeled.ArrayLike,
            width_yx: kgpy.labeled.ArrayLike,
            unit: u.Unit,
    ):
        a = self._calc_matrix(xy, yx, width_xy, width_yx, unit)
        b = a.inverse_numpy()
        c = b.inverse_numpy()
        assert np.all(np.isclose(a @ b, kgpy.matrix.Cartesian2D.identity()))
        assert np.all(np.isclose(c, a))
        assert np.all(np.isclose(~a.transpose, b.transpose))

    def test__invert__(
            self,
            xy: kgpy.labeled.ArrayLike,
            yx: kgpy.labeled.ArrayLike,
            width_xy: kgpy.labeled.ArrayLike,
            width_yx: kgpy.labeled.ArrayLike,
            unit: u.Unit,
    ):
        a = self._calc_matrix(xy, yx, width_xy, width_yx, unit)
        b = ~a
        c = ~b
        assert np.all(np.isclose(a @ b, kgpy.matrix.Cartesian2D.identity()))
        assert np.all(np.isclose(c, a))
        assert np.all(np.isclose(~a.transpose, b.transpose))

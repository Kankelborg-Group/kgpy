import pytest
import kgpy.vector
import kgpy.matrix

identity = kgpy.matrix.Cartesian2D.identity()
xx = kgpy.matrix.Cartesian2D(
    x=kgpy.vector.Cartesian2D(x=1),
    y=kgpy.vector.Cartesian2D(),
)
xy = kgpy.matrix.Cartesian2D(
    x=kgpy.vector.Cartesian2D(y=1),
    y=kgpy.vector.Cartesian2D(),
)
yx = kgpy.matrix.Cartesian2D(
    x=kgpy.vector.Cartesian2D(),
    y=kgpy.vector.Cartesian2D(x=1),
)
test_1 = kgpy.matrix.Cartesian2D(
    x=kgpy.vector.Cartesian2D(x=1, y=2),
    y=kgpy.vector.Cartesian2D(x=3, y=4),
)
test_2 = kgpy.matrix.Cartesian2D(
    x=kgpy.vector.Cartesian2D(x=5, y=6),
    y=kgpy.vector.Cartesian2D(x=7, y=8),
)


class TestCartesian2D:

    @pytest.mark.parametrize(
        argnames='a',
        argvalues=[test_1, test_2],
    )
    def test_transpose(self, a: kgpy.matrix.Cartesian2D):
        assert a == a.transpose.transpose

    @pytest.mark.parametrize(
        argnames='a,b,out',
        argvalues=[
            (identity, kgpy.vector.Cartesian2D.x_hat(), kgpy.vector.Cartesian2D.x_hat()),
            (identity, identity, identity, ),
            (xy, yx, xx),
        ]
    )
    def test__matmul__(
            self,
            a: kgpy.matrix.Cartesian2D,
            b: kgpy.matrix.Cartesian2D,
            out: kgpy.matrix.Cartesian2D,
    ):
        out_test = a @ b
        assert isinstance(out_test, type(out))
        assert out_test == out

    @pytest.mark.parametrize(
        argnames='a',
        argvalues=[test_1, test_2],
    )
    def test__invert__(self, a: kgpy.matrix.Cartesian2D):
        b = ~a
        c = ~b
        assert a @ b == identity
        assert c == a
        assert ~a.transpose == b.transpose

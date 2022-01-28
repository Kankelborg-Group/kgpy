import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vector


class TestCartesian2D:

    @pytest.mark.parametrize('a', [
        kgpy.vector.Cartesian2D(2, 3),
        kgpy.vector.Cartesian2D(2, 3) * u.mm,
    ])
    @pytest.mark.parametrize('b', [
        4,
        4 * u.mm,
        kgpy.labeled.LinearSpace(4, 5, 11, axis='b'),
        kgpy.labeled.LinearSpace(4, 5, 11, axis='b') * u.mm,
        kgpy.uncertainty.Uniform(4, 2),
        kgpy.vector.Cartesian2D(3, 4),
    ])
    def test__mul__(self, a, b):
        c = a * b
        d = b * a
        assert isinstance(c, kgpy.vector.Cartesian2D)
        assert isinstance(d, kgpy.vector.Cartesian2D)
        assert np.all(c == d)
        if not isinstance(b, kgpy.vector.Cartesian2D):
            assert np.all(c.x == a.x * b)
            assert np.all(c.y == a.y * b)
        else:
            assert np.all(c.x == a.x * b.x)
            assert np.all(c.y == a.y * b.y)

    @pytest.mark.parametrize(
        argnames='x',
        argvalues=[
            kgpy.labeled.LinearSpace(0, 1, num=5, axis='a'),
            kgpy.uncertainty.Normal(kgpy.labeled.LinearSpace(0, 1, num=5, axis='a'), width=0.1)
        ]
    )
    @pytest.mark.parametrize(
        argnames='y',
        argvalues=[
            0,
            kgpy.labeled.LinearSpace(0, 1, num=5, axis='a'),
            kgpy.labeled.LinearSpace(0, 1, num=5, axis='a') * kgpy.labeled.LinearSpace(0, 1, num=3, axis='b'),
            kgpy.uncertainty.Normal(kgpy.labeled.LinearSpace(0, 1, num=5, axis='a'), width=0.1)
        ]
    )
    def test_broadcast_to(self, x, y):
        a = kgpy.vector.Cartesian2D(x, y)
        b = np.broadcast_to(a, a.shape)
        assert b.x.shape == a.shape
        assert b.y.shape == a.shape


    @pytest.mark.parametrize(
        argnames='x',
        argvalues=[
            kgpy.labeled.LinearSpace(0, 1, num=5, axis='a'),
            kgpy.uncertainty.Normal(kgpy.labeled.LinearSpace(0, 1, num=5, axis='a'), width=0.1)
        ]
    )
    @pytest.mark.parametrize(
        argnames='y, color',
        argvalues=[
            (kgpy.labeled.LinearSpace(0, 1, num=5, axis='a'), 'black'),
            (
                kgpy.labeled.LinearSpace(0, 1, num=5, axis='a') * kgpy.labeled.LinearSpace(0, 1, num=3, axis='b'),
                kgpy.labeled.Array(np.array(['black', 'blue', 'red',]), axes=['b']),
            ),
            (kgpy.uncertainty.Normal(kgpy.labeled.LinearSpace(0, 1, num=5, axis='a'), width=0.1), 'black'),
        ]
    )
    def test_plot(self, x, y, color):
        fig, ax = plt.subplots()
        a = kgpy.vector.Cartesian2D(x, y)
        lines = a.plot(ax, axis_plot='a', color=color)
        assert lines
        # plt.show()

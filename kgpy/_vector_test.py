import pytest
import numpy as np
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

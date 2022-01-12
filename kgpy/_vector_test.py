import pytest
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
    ])
    def test__mul__(self, a, b):
        c = a * b
        d = b * a
        assert isinstance(c, kgpy.vector.AbstractVector)
        assert isinstance(d, kgpy.vector.AbstractVector)

        print(c)
        print(d)
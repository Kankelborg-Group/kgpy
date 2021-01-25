import pytest
import astropy.units as u
from . import units


@pytest.fixture
def tq1():
    return units.TolQuantity(10 * u.mm, tmin=-1, tmax=1)


@pytest.fixture
def tq2():
    return units.TolQuantity(20 * u.mm, tmin=-1, tmax=2)


class TestTolQuantity:

    def test__add__(self, tq1: units.TolQuantity, tq2: units.TolQuantity):
        tq3 = tq1 + tq2
        assert tq3.value == (tq1 + tq2).value
        assert tq3.tmin == tq1.tmin + tq2.tmin
        assert tq3.tmax == tq1.tmax + tq2.tmax

    def test__mul__(self, tq1: units.TolQuantity, tq2: units.TolQuantity):
        tq3 = tq1 * tq2
        assert tq3.value == (tq1 * tq2).value
        assert tq3.amin == tq1.amin * tq2.amin
        assert tq3.amax == tq1.amax * tq2.amax




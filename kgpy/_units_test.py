import pytest
import astropy.units as u
from . import units


@pytest.fixture
def tq1():
    return units.TolQuantity(10 * u.mm, vmin=-1 * u.mm, vmax=1 * u.mm)


@pytest.fixture
def tq2():
    return units.TolQuantity(20 * u.mm, vmin=-1 * u.mm, vmax=2 * u.mm)


class TestTolQuantity:

    def test__add__(self, tq1: units.TolQuantity, tq2: units.TolQuantity):
        tq3 = tq1 + tq2
        assert tq3.value == (tq1 + tq2).value
        assert tq3.vmin == tq1.vmin + tq2.vmin
        assert tq3.vmax == tq1.vmax + tq2.vmax

    def test__mul__(self, tq1: units.TolQuantity, tq2: units.TolQuantity):
        tq3 = tq1 * tq2
        assert tq3.value == (tq1 * tq2).value
        assert tq3.vmin == tq1.vmin * tq2.vmin
        assert tq3.vmax == tq1.vmax * tq2.vmax




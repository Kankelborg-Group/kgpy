import astropy.units as u

from . import CoordinateBreak


class TestCoordinateBreak:

    def test__init__(self):

        c = CoordinateBreak()

        assert isinstance(c.decenter, u.Quantity)
        assert c.decenter.unit.is_equivalent(u.m)

        assert isinstance(c.tilt, u.Quantity)
        assert c.tilt.unit.is_equivalent(u.deg)

    def test_config_broadcast(self):

        tilt = [[0, 0, 0]] * u.deg

        c = CoordinateBreak(tilt=tilt)

        assert c.config_broadcast.shape == tilt.shape[:~0]


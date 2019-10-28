import astropy.units as u

from . import aperture


class TestRectangular:

    def test_points(self):

        r = aperture.Rectangular()

        assert isinstance(r.points, u.Quantity)


class TestRegularPolygon:

    def test_points(self):

        p = aperture.Octagon()

        assert isinstance(p.points, u.Quantity)

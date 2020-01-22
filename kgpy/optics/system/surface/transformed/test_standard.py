import astropy.units as u

from kgpy.optics.system import surface

from . import standard


class TestStandard:

    def test_aperture_surface(self):

        s = standard.Standard(
            name='test',
            mechanical_aperture=surface.aperture.Rectangular()
        )

        print(s.aperture_surface)
import nptyping as npt
import astropy.units as u

from . import Standard, Material, Aperture


class TestStandard:

    def test__init__(self):

        s = Standard()

        assert isinstance(s.radius, u.Quantity)
        assert isinstance(s.conic, u.Quantity)
        assert s.material is None or isinstance(s.material, Material) or isinstance(s.material, npt.Array[Material])

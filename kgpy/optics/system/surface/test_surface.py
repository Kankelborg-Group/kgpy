import typing as tp
import nptyping as npt
import astropy.units as u

from . import Surface


class TestSurface:

    def test__init__(self):

        s = Surface()

        assert isinstance(s.name, str) or isinstance(s.name, npt.Array[str])
        assert isinstance(s.is_stop, bool) or isinstance(s.is_stop, npt.Array[bool])
        assert isinstance(s.thickness, u.Quantity)


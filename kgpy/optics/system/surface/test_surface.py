import pytest
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

    @pytest.mark.parametrize('test_surface, shape', [
        (Surface(), ()),
        (Surface(thickness=[0, 1, 2] * u.m), (3,)),
    ])
    def test_config_broadcast(self, test_surface, shape):

        assert test_surface.config_broadcast.shape == shape


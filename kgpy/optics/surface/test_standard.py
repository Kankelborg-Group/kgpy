import pytest
import typing as tp
import nptyping as npt
import astropy.units as u

from . import Standard, Material, Aperture


class TestStandard:

    def test__init__(self):

        s = Standard()

        assert isinstance(s.radius, u.Quantity)
        assert isinstance(s.conic, u.Quantity)
        assert s.material is None or isinstance(s.material, Material) or isinstance(s.material, npt.Array[Material])

    @pytest.mark.parametrize('test_surface, expected_shape', [
        (Standard(), ()),
        (Standard(thickness=[0, 1, 2] * u.m), (3,)),
        (Standard(radius=[1, 2, 3] * u.m), (3,)),
        (Standard(decenter_before=[[0, 0, 0], [1, 0, 0]] * u.m), (2,)),
        (Standard(thickness=[0, 1, 2] * u.m, radius=[[0], [1], [2]] * u.m), (3, 3))
    ])
    def test_config_broadcast(self, test_surface, expected_shape):

        assert test_surface.config_broadcast.shape == expected_shape

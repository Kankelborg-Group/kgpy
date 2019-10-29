import pytest
import numpy as np
import astropy.units as u

from . import System, Fields, Wavelengths, surface


class TestSystem:

    @pytest.mark.parametrize('test_system, expected_shape', [
        (System(name='', surfaces=[surface.Standard()], fields=Fields(), wavelengths=Wavelengths()), ()),
        (System(name='', surfaces=[
            surface.Standard(thickness=[0, 1, 2] * u.m),
            surface.Standard(radius=[0, 1, 2] * u.mm),
        ], fields=Fields(), wavelengths=Wavelengths()), (3,)),
        (System(name='', surfaces=[
            surface.Standard(thickness=[0, 1, 2] * u.m),
            surface.Standard(radius=[[0], [1], [2]] * u.mm),
        ], fields=Fields(), wavelengths=Wavelengths()), (3, 3)),
    ])
    def test_broadcastable_attrs(self, test_system, expected_shape):

        assert test_system.broadcasted_attrs.shape == expected_shape




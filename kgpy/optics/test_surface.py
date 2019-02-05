
import pytest
from numbers import Real
from typing import Union
import astropy.units as u

from kgpy.optics import Surface


class TestSurface:

    def test__str__(self):

        surf = Surface('test')

        assert isinstance(surf.__str__(), str)

    @pytest.mark.parametrize('unit', [1, u.s, u.m / u.s, u.dimensionless_unscaled])
    def test__init__(self, unit: Union[Real, u.Unit]):
        """
        Check that the Constructor will only accept thicknesses with dimensions of length
        :return: None
        """

        # Check that providing a thickness with units of seconds that a TypeError is raised
        with pytest.raises(TypeError):
            Surface('test', thickness=1 * unit)

    def test_cs_break(self):
        """
        Test the coordinate break feature of the Surface class.
        This test defines four surfaces, each with a 90-degree rotation.
        If the test is successful, the back face of the last surface should be at the origin.
        :return: None
        """

        # Give each surface some arbitrary thickness
        t = 1 * u.mm

        # Define


        s1 = Surface('Surface 1', thickness=t, )
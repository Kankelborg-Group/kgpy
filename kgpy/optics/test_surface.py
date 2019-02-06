
import pytest
from numbers import Real
from typing import Union
import numpy as np
import astropy.units as u
import quaternion as q

from kgpy.math import Vector, CoordinateSystem
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

        print(type(u.mm))

        # Give each surface some arbitrary thickness
        t = 1 * u.mm

        # Define a 90-degree coordinate break
        X = Vector([0, 0, 0] * u.mm)
        Q = q.from_euler_angles(np.pi/2, 0, 0)
        cs = CoordinateSystem(X, Q)

        # Define the four test surfaces to arrange into a square
        s1 = Surface('Surface 1', thickness=t, cs_break=cs)
        s2 = Surface('Surface 2', thickness=t, cs_break=cs)
        s3 = Surface('Surface 3', thickness=t, cs_break=cs)
        s4 = Surface('Surface 4', thickness=t, cs_break=cs)

        # Link each surface properly
        s2.previous_surf = s1
        s3.previous_surf = s2
        s4.previous_surf = s3

        print(s1.cs)
        print(s2.cs)
        print(s3.cs)
        print(s4.cs)
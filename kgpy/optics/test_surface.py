
import pytest
from numbers import Real
from typing import Union
import numpy as np
import astropy.units as u
import quaternion as q

from kgpy.math import Vector, CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
from kgpy.optics import Surface


class TestSurface:

    @pytest.mark.parametrize('unit', [1, u.s, u.m / u.s, u.dimensionless_unscaled])
    def test__init__(self, unit: Union[Real, u.Unit]):
        """
        Check that the Constructor will only accept thicknesses with dimensions of length
        :return: None
        """

        # Check that providing a thickness with units of seconds that a TypeError is raised
        with pytest.raises(TypeError):
            Surface('test', thickness=1 * unit)

    def test_cs(self):
        """
        Test the coordinate break feature of the Surface class.
        This test defines four surfaces, each with a 90-degree rotation.
        If the test is successful, the back face of the last surface should be at the origin.
        :return: None
        """

        # Give each surface some arbitrary thickness
        t = 1 * u.mm

        # Define a 90-degree coordinate break
        X = Vector([0, 0, 0] * u.mm)
        Q = q.from_euler_angles(0, np.pi/2, 0)
        cs = CoordinateSystem(X, Q)

        # Define the four test surfaces to arrange into a square
        s1 = Surface('Surface 1', thickness=t)
        s2 = Surface('Surface 2', thickness=t, cs_break=cs)
        s3 = Surface('Surface 3', thickness=t, cs_break=cs)
        s4 = Surface('Surface 4', thickness=t, cs_break=cs)

        # Link each surface properly
        s2.previous_surf = s1
        s3.previous_surf = s2
        s4.previous_surf = s3

        assert s4.back_cs.X.isclose(X)

    def test_front_cs(self):
        """
        Test the front coordinate system.
        Two surfaces, one with no tilt/dec and one with tilt/dec.
        Check that Surface 2 is rotated relative to Surface 1 and also that the thickness vector of Surface 2 is
        parallel to Surface 1.
        :return: None
        """

        # Create test coordinate system
        X = Vector([0, 0, 0] * u.mm)
        Q = q.from_euler_angles(0, np.pi/2, 0)
        cs = CoordinateSystem(X, Q)

        # Create two test surfaces, the second will have the nonzero tilt/dec system
        t = 1*u.mm
        s1 = Surface('s1', thickness=t, cs_break=cs)
        s2 = Surface('s2', thickness=t, tilt_dec=cs)

        # link up the surfaces
        s2.previous_surf = s1

        # Check that the front coordinate system has rotated the full 180 degrees
        assert np.isclose(s2.front_cs.Q, q.from_euler_angles(0, np.pi, 0))

        # Check that the thickness vector points in the x-hat direction
        assert s2.back_cs.isclose(cs + Vector([2, 0, 0] * u.mm))

    def test_back_cs(self):
        """
        Check that the back surface is computed properly.
        :return: None
        """

        # Create test coordinate system
        x = 1
        a = np.pi/4
        X = Vector([x, x, x] * u.mm)
        Q = q.from_euler_angles(a, np.arccos(1 * u.mm / X.mag), 0)
        cs = CoordinateSystem(X, Q)

        # Create test surface
        s = Surface('s', cs_break=cs, thickness=X.mag)

        # Check that the back surface is two millimeters from the origin in each axis
        assert s.back_cs.isclose(cs + X)

    def test__str__(self):

        surf = Surface('test')

        assert isinstance(surf.__str__(), str)

import pytest
from numbers import Real
from typing import Union
import numpy as np
import astropy.units as u
import quaternion as q
from copy import deepcopy

from kgpy.math import Vector, CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
from kgpy.optics import Surface, Component, System


class TestSurface:

    @pytest.mark.parametrize('unit', [1, u.s, u.m / u.s, u.dimensionless_unscaled])
    def test_thickness(self, unit: Union[Real, u.Unit]):
        """
        Check that the Constructor will only accept thicknesses with dimensions of length
        :return: None
        """

        # Check that providing a thickness with units of seconds that a TypeError is raised
        with pytest.raises(TypeError):
            Surface('test', thickness=1 * unit)

    def test_system_index(self):
        """
        Check that the system index is being calculated correctly by adding several surfaces and checking the indices.
        :return: None
        """

        # Give each surface some arbitrary thickness
        t = 1 * u.mm

        # Define three test surfaces
        s1 = Surface('s1', thickness=t)
        s2 = Surface('s2', thickness=t)
        s3 = Surface('s3', thickness=t)

        # Add the three test surfaces to a system
        sys = System('sys')
        sys.append_surface(s1)
        sys.append_surface(s2)
        sys.append_surface(s3)

        # Check that the indices are calculated correctly
        assert s1.system_index is 0
        assert s2.system_index is 1
        assert s3.system_index is 2

    def test_component_index(self):
        """
        Check that the component index is being calculated correctly by adding several surfaces and checking the
        indices.
        :return: None
        """

        # Give each surface some arbitrary thickness
        t = 1 * u.mm

        # Define three test surfaces
        s1 = Surface('s1', thickness=t)
        s2 = Surface('s2', thickness=t)
        s3 = Surface('s3', thickness=t)

        # Add the three test surfaces to a component
        c = Component('c')
        c.append_surface(s1)
        c.append_surface(s2)
        c.append_surface(s3)

        # Check that the indices are calculated correctly
        assert s1.component_index is 0
        assert s2.component_index is 1
        assert s3.component_index is 2

    def test_T(self):
        """
        Test that the thickness vector is pointing in the direction of global z-hat by default
        :return: None
        """

        # Create test surface
        s1 = Surface('s1', thickness=1*u.mm)

        # Check that the thickness vector is in the global z-hat direction
        assert s1.T == Vector([0, 0, 1] * u.mm)

        # Check that the thickness vector is not in the global x-hat direction
        assert s1.T != Vector([1, 0, 0] * u.mm)

    def test_is_object(self):

        # Create a new test system
        s = System('s')

        # Check that the first surface in the system reports that it is the object surface
        assert s.obj_surface.is_object

    def test_previous_cs(self):

        # Give each surface some arbitrary thickness
        t = 1 * u.mm

        # Define three test surfaces
        s1 = Surface('s1', thickness=t)
        s2 = Surface('s2', thickness=t)
        s3 = Surface('s3', thickness=t)

        # Check that the surfaces return the global coordinate system by default
        assert s1._previous_cs == gcs()

        # Add the three surfaces to two components, with the second component getting two surfaces
        cs = gcs() * q.from_euler_angles(0, np.pi/2, 0)
        c1 = Component('c1')
        c2 = Component('c2', cs_break=cs)
        c1.append_surface(s1)
        c2.append_surface(s2)
        c2.append_surface(s3)

        # Check that the coordinate systems are correct for independent components
        assert s1._previous_cs.isclose(gcs())
        assert s2._previous_cs.isclose(cs)
        assert s3._previous_cs.isclose(cs + (t * cs.xh_g))

        # Place the components into a test system
        sys = System('sys')
        sys.append_component(c1)
        sys.append_component(c2)

        # Check that the coordinate systems are correct for the two components appended together
        assert s1._previous_cs.isclose(gcs())
        assert s2._previous_cs.isclose(cs + t * cs.zh_g)
        assert s3._previous_cs.isclose(cs + t * cs.zh_g + t * cs.xh_g)

    def test_cs(self):
        """
        Test the front coordinate system.
        Two surfaces, one with no tilt/dec and one with tilt/dec.
        Check that Surface 2 is rotated relative to Surface 1 and also that the thickness vector of Surface 2 is
        parallel to Surface 1.
        :return: None
        """

        # Create test coordinate system
        X = Vector([0, 0, 0] * u.mm)
        Q = q.from_euler_angles(0, np.pi / 2, 0)
        cs = CoordinateSystem(X, Q)

        # Create two test surfaces, the second will have the nonzero tilt/dec system
        t = 1 * u.mm
        s1 = Surface('s1', thickness=t, cs_break=cs)
        s2 = Surface('s2', thickness=t, tilt_dec=cs)

        # Add the two test surfaces to a component
        c = Component('c')
        c.append_surface(s1)
        c.append_surface(s2)

        # Check that the front coordinate system has rotated the full 180 degrees
        assert np.isclose(s2.cs.Q, q.from_euler_angles(0, np.pi, 0))

        # Check that the thickness vector points in the x-hat direction
        assert s2.back_cs.isclose(cs + Vector([2, 0, 0] * u.mm))

    def test_front_cs(self):
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

        # Add the test surfaces to a component
        c = Component('c')
        c.append_surface(s1)
        c.append_surface(s2)
        c.append_surface(s3)
        c.append_surface(s4)

        # Check that the translation vector of the last surface is close to the origin
        assert s4.back_cs.X.isclose(X)

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

    def test__eq__(self):

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

        # Add the test surfaces to a component
        c1 = Component('c1')
        c1.append_surface(s1)
        c1.append_surface(s2)
        c1.append_surface(s3)
        c1.append_surface(s4)

        # Make a deepcopy of the component for testing
        c2 = deepcopy(c1)

        # Check for equality between the copies
        assert c1.surfaces[0] == c2.surfaces[0]
        assert c1.surfaces[1] == c2.surfaces[1]
        assert c1.surfaces[2] == c2.surfaces[2]
        assert c1.surfaces[3] == c2.surfaces[3]

        # Check that the first surface is not equal to the other three
        assert c1.surfaces[0] != c1.surfaces[1]
        assert c1.surfaces[0] != c1.surfaces[2]
        assert c1.surfaces[0] != c1.surfaces[3]

        # Modify some parameters in the copy, and check that the surfaces are not equal
        c2.surfaces[0].name = 'foo'
        assert c1.surfaces[0] != c2.surfaces[0]

    def test__str__(self):

        surf = Surface('test')

        assert isinstance(surf.__str__(), str)

import pytest
import numpy as np
import astropy.units as u
from quaternion import from_euler_angles as euler

from kgpy.optics import Surface, Component, System
from kgpy.math import Vector
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs

class TestSystem:

    def test_surfaces(self):

        # Give each surface some arbitrary thickness
        t = 1 * u.mm

        # Construct the first component
        c1 = Component('c1')
        s1 = Surface('s1', thickness=t)
        c1.append_surface(s1)

        # Construct the second component
        c2 = Component('c2')
        s2 = Surface('s2', thickness=t)
        c2.append_surface(s2)

        # Create a new optical system and append both components
        sys = System('sys')
        sys.append_component(c1)
        sys.append_component(c2)

        # Check that the surfaces were linked up correctly
        assert sys.surfaces == [s1, s2]

    def test_append_component(self):

        # Give each surface some arbitrary thickness
        t = 1 * u.mm

        # Construct the first component
        c1 = Component('c1')
        s1 = Surface('s1', thickness=t)
        c1.append_surface(s1)

        # Construct the second component
        c2 = Component('c2')
        s2 = Surface('s2', thickness=t)
        c2.append_surface(s2)

        # Create a new optical system and append both components
        sys = System('sys')
        sys.append_component(c1)
        sys.append_component(c2)

        # Check that the surfaces were linked up correctly
        assert sys.first_surface == s1
        assert s1.next_surf_in_system == s2
        assert s2.prev_surf_in_system == s1

    @pytest.mark.parametrize('b', (0, np.pi/2))
    def test_calc_surface_intersections(self, b: float):

        Q = euler(0, b, 0)
        cs = gcs() * Q

        # Give each surface some arbitrary thickness
        t = 1 * u.mm

        # Construct the first component
        c1 = Component('c1', cs_break=cs)
        s1 = Surface('s1', thickness=t)
        s2 = Surface('s2', thickness=t)
        c1.append_surface(s1)
        c1.append_surface(s2)

        # Construct the second component
        c2 = Component('c2')
        s3 = Surface('s3', thickness=-t)
        s4 = Surface('s4', thickness=-t)
        c2.append_surface(s3)
        c2.append_surface(s4)

        # Create a new optical system and append both components
        sys = System('sys')
        sys.append_component(c1)
        sys.append_component(c2)

        print(sys)

        # Create coordinate system for the location of the test surface
        bcs = gcs() + Vector([0, 0, 0.5] * u.mm)

        surfaces = sys.calc_surface_intersections(bcs)

        assert surfaces == [s1, s4]



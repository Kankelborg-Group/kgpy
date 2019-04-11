
import pytest
from copy import deepcopy
import numpy as np
import astropy.units as u
from quaternion import from_euler_angles as euler

from kgpy.optics import Surface, Component, System
from kgpy.math import Vector
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs

__all__ = ['system', 'TestSystem']

b = (0, np.pi / 4, np.pi / 2)


@pytest.fixture(params=b)
def system(request):
    # Define parametrized coordinate system
    Q = euler(0, request.param, 0)
    cs = gcs() * Q

    # Give each surface some arbitrary thickness
    t = 1 * u.mm    # type: u.Quantity

    # Construct the first component
    c1 = Component('c1', cs_break=cs)
    s1 = Surface('s1', thickness=t)
    s2 = Surface('s2', thickness=t)
    c1.append(s1)
    c1.append(s2)

    # Construct the second component
    c2 = Component('c2')
    s3 = Surface('s3', thickness=-t)
    s4 = Surface('s4', thickness=-t)
    c2.append(s3)
    c2.append(s4)

    # Create a new optical system and append both components
    sys = System('sys')

    # Insert all surfaces before the image surface
    sys.insert(s1, -1)
    sys.insert(s2, -1)
    sys.insert(s3, -1)
    sys.insert(s4, -1)

    return sys


class TestSystem:

    def test_obj_surface(self, system):

        # Check that the object surface is an instance of the surface class
        assert isinstance(system.object, Surface)

        # Check that the object surface has the appropriate name string
        assert system.object.name == system.object_str

        # Check that the surface reports that it is the object.
        assert system.object.is_object

    def test_surfaces(self, system):

        # Check that the surfaces attribute is populated
        assert system._surfaces

        # For every surface in the list, check that it is a valid surface
        for surf in system._surfaces:

            # Check that each element is an instance of the Surface class
            assert isinstance(surf, Surface)

    def test_components(self, system):

        # Check that the components attribute is populated
        assert system.components

        # For every component in the list, check that it is a valid component
        for name, comp in system.components.items():

            # Check that the key is a string
            assert isinstance(name, str)

            # Check that the value is a Component
            assert isinstance(comp, Component)

    @pytest.mark.parametrize('i', (0, 1, 2, -1, -2))
    def test_insert_surface(self, system, i: int):
        # Todo: more robust testing for negative indices, check actual coordinate systems

        # Create new test surface to insert
        s_test = Surface('s_test')

        # Insert the test surface into the system
        system.insert(s_test, i)

        # Check that the pointer to the system was set correctly
        assert s_test.sys == system

        # If the index is greater than zero, we expect the test surface to be at the same index within the surface list.
        if i >= 0:
            assert system._surfaces[i] == s_test

        # Otherwise, the index is less than zero, and the test surface is one index before the insertion index.
        # This is because index=-1 doesn't refer to the same element in the new and old list, i.e. the insert function
        # places the element before the list.
        else:
            assert system._surfaces[i - 1] == s_test

    def test_append_surface(self, system):

        # Create new test surface to insert
        s_test = Surface('s_test')

        # Insert the test surface into the system
        system.append(s_test)

        # Check that the pointer to the system was set correctly
        assert s_test.sys == system

        # Check that the test surface is the last surface in the system.
        assert system._surfaces[-1] == s_test

    def test_append_component(self, system):

        # Construct test component
        c_test = Component('c_test')
        s_test = Surface('s_test')
        c_test.append(s_test)

        # Append component to system
        system.insert_component(c_test)

        # Check that the component is in the system components dict.
        assert system.components[c_test.name] == c_test

        # Check that the test surface is the last surface in the system
        assert system._surfaces[-1] == s_test

    def test_add_baffle(self, system):

        # Create a copy of the system for later comparison
        old_sys = deepcopy(system)

        # Create coordinate system for the location of the test surface
        bcs = gcs() + Vector([0, 0, 0.5] * u.mm)

        # Add a a baffle at the coordinate system specified by bcs
        baffle = system.add_baffle('b1', baffle_cs=bcs)

        # Check to see if a baffle was added for this angle of b
        if baffle.first_surface is not None:

            # Check that all the original surfaces have not moved, and that the two new surfaces are at the location we
            # specified.
            assert system._surfaces[0].cs.isclose(old_sys._surfaces[0].cs)
            assert system._surfaces[1].cs.isclose(old_sys._surfaces[1].cs)
            assert system._surfaces[2].cs.isclose(old_sys._surfaces[2].cs)
            assert system._surfaces[3].cs.isclose(bcs)
            assert system._surfaces[4].cs.isclose(old_sys._surfaces[3].cs)
            assert system._surfaces[5].cs.isclose(old_sys._surfaces[4].cs)
            assert system._surfaces[6].cs.isclose(old_sys._surfaces[5].cs)
            assert system._surfaces[7].cs.isclose(bcs)
            assert system._surfaces[8].cs.isclose(old_sys._surfaces[6].cs)

        else:

            assert len(baffle._surfaces) == 0

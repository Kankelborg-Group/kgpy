
import pytest
import astropy.units as u

from kgpy.optics import Surface, Component
from kgpy.math.coordinate_system import GlobalCoordinateSystem

__all__ = ['TestComponent']


class TestComponent:

    t = [0, 1]

    def test__init__(self):

        # Create a test component using the default settings
        name = 'test'
        c = Component(name)

        # Check initial properties of the object
        assert c.name == name   # Name equal to the name we set it to
        assert c.comment == ''  # Default comment is the empty string
        assert c.first_surface is None
        assert not c.surfaces   # The list of surfaces should be empty to start

    def test__init__matching(self):

        # Create a new component to test the matching surface feature
        name = 'test'
        c = Component(name, matching_surf=True)

        # Check that the matching surface has been created as expected
        assert c.first_surface is not None
        assert c.first_surface.name == name

    def test_surfaces(self):

        # Define test surfaces
        t = 1 * u.mm
        s1 = Surface('s1', thickness=t)
        s2 = Surface('s2', thickness=t)
        s3 = Surface('s3', thickness=t)

        # Construct component out of surfaces
        c = Component('c')
        c.append_surface(s1)
        c.append_surface(s2)
        c.append_surface(s3)

        # Check that the list of surfaces is returned correctly
        assert c.surfaces == [s1, s2, s3]

    @pytest.mark.parametrize('t1', t)
    @pytest.mark.parametrize('t2', t)
    def test_T(self, t1: u.Quantity, t2: u.Quantity):

        # Define two test surfaces and an empty component
        s1 = Surface('Surface 1', thickness=t1*u.mm)
        s2 = Surface('Surface 2', thickness=t2*u.mm)

        # Add the two surfaces to an empty Component
        c = Component('test')
        c.append_surface(s1)
        c.append_surface(s2)

        # Check that the thickness of the component is equal to the sum of the surface thicknesses
        assert c.T.mag == (t1 + t2) * u.mm

    @pytest.mark.parametrize('t1', t)
    @pytest.mark.parametrize('t2', t)
    def test_append_surface(self, t1: u.Quantity, t2: u.Quantity):

        # Define two test surfaces
        s1 = Surface('Surface 1', thickness=t1*u.mm)
        s2 = Surface('Surface 2', thickness=t2*u.mm)

        # Create test Component
        c = Component('test')

        # Add the first surface to the component
        c.append_surface(s1)

        # Check that the translation of the surface is zero
        assert c.surfaces[0].cs.X == 0

        # Add the second surface to the component
        c.append_surface(s2)

        # Check that the z-translation of the surface is equal to the thickness of the first surface
        assert c.surfaces[1].cs.X.z == t1*u.mm

    def test__str__(self):

        # Create default test Component
        c = Component('test')

        # Test that the return value is of type string
        assert isinstance(c.__str__(), str)
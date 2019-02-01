
from unittest import TestCase
import astropy.units as u

from kgpy.optics import Surface, Component

__all__ = ['TestComponent']


class TestComponent(TestCase):

    def setUp(self):

        # Define two test surfaces
        s1 = Surface('Surface 1', thickness=1.0 * u.mm)
        s2 = Surface('Surface 2', thickness=2.0 * u.mm)

        # Create a component out of the two test surfaces
        self.comp = Component('test', [s1, s2])

    def test__init__(self):

        # Test that the origin of Surface 2 is moved forward by a millimeter
        self.assertEqual(self.comp.surfaces[-1].cs.X.z, 2.0 * u.mm)

    def test__str__(self):

        # Test that the return value is of type string
        self.assertTrue(isinstance(self.comp.__str__(), str))

    def test_T(self):

        print(self.comp.T)

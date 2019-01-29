
from unittest import TestCase
import astropy.units as u

from kgpy.optics import Surface


class TestSurface(TestCase):

    def setUp(self):

        self.surf = Surface('test')

    def test__str__(self):

        self.assertIsInstance(self.surf.__str__(), str)

    def test_thickness_unit(self):
        """
        Check that the Constructor will only accept thicknesses with dimensions of length
        :return: None
        """

        # Check that providing a thickness with units of seconds that a TypeError is raised
        with self.assertRaises(TypeError):
            Surface('test', thickness=1 * u.s)


from unittest import TestCase

from . import Zemax
from .zemax.zemax import TestZemax


class Surface:
    """
    This class represents a single Zemax surface, exists to allow a few extra properties to be associated with the
    surface.
    """

    def __init__(self, zmx, name, comment):
        """
        Constructor for class surface. Takes a comment and matches it to a surface in Zemax.

        :param zmx: Pointer to Zemax design instance
        :param name: Human-readable name of surface
        :param comment: Comment string to match against list of Zemax surfaces. First surface matching this comment will
        be saved.
        """

        # Save arguments to class variables
        self.name = name
        self.comment = comment

        # Open Zemax surface object
        self.zmx_surf = zmx.find_surface(self.comment)


class TestSurface(TestCase):
    """
    Test a basic surface against the test Zemax design.
    """

    def setUp(self):
        """
        Initialize Zemax design object and surface
        :return: None
        """

        self.zmx = Zemax(TestZemax.test_path)
        self.surf = Surface(self.zmx, 'primary', 'Primary')

    def test__init(self):
        """
        Test constructor. Check that the loaded surface is not None
        :return: None
        """

        self.assertTrue(self.surf.zmx_surf is not None)





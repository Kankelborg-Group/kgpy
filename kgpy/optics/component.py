
from abc import ABC, abstractmethod
from unittest import TestCase

from . import Surface, Zemax
from .zemax.zemax import TestZemax

__all__ = ['Component']


class Component(ABC):
    """
    An optical component is a collection of one or more surfaces such as the tips/tilts, clear aperture, and the optical
    surface of an optical element.
    """

    @abstractmethod
    def __init__(self, zmx):
        """
        Abstract constructor for objects of class Component. This constructor is not designed to be called directly, it
        depends on the instantiation of the name and surface_comment class members

        :param zmx: Pointer to Zemax object representing a particular optical design.
        """

        # Save pointer to Zemax object
        self.zmx = zmx

        # Allocate empty dictionary for Surface objects
        self._surfaces = {}

        # Loop through every name/comment pair in provided dictionary to initialize surfaces
        for name, comment in self.surface_comment_dict.items():
            s = Surface(zmx, name, comment)
            self._surfaces[name] = s

    @property
    @abstractmethod
    def name(self):
        """
        This method needs to be abstract since subclasses will declare there name here

        :return: Human-readable name of the optical component
        :rtype: str
        """
        return 'default name'

    @property
    @abstractmethod
    def surface_comment_dict(self):
        """
        Dictionary of comments representing the surfaces we want to match against.

        :return: Dictionary, where every key/value pair is the name of the surface and the comment associated with the
        surface, respectively.
        :rtype: dict
        """
        return {'default key': 'default value'}

    @property
    def surfaces(self):
        """
        Dictionary of surfaces representing the optical component

        :return: Dictionary, where every key/value pair is the name of the surface and a pointer to the surface object
        :rtype: dict
        """
        return self._surfaces


class DefaultPrimary(Component):
    """
    This concrete component loads the primary mirror of the test Zemax file as a component.
    """

    def __init__(self, zmx):

        super().__init__(zmx)

    @property
    def name(self):
        return 'Primary'

    @property
    def surface_comment_dict(self):
        return {
            'primary': 'Primary'
        }


class TestComponent(TestCase):
    """
    This class is designed to test the abstract methods of the Component class.
    Testing an abstract class is accomplished through defining a concrete example of the Component class.
    This class uses the `./zemax/test_model.zmx` Zemax file to exercise the methods
    """

    def setUp(self):
        """
        Set up the test by loading the test Zemax design
        :return: None
        """

        # Instantiate Zemax design for use by other tests.
        self.zmx = Zemax(TestZemax.test_path)

        # Initialize concrete component for testing
        self.component = DefaultPrimary(self.zmx)

    def test__init__(self):
        """
        Test the constructor by defining a concrete component and checking that every surface in that component is not
        None.
        :return: None
        """

        # Loop through every surface and check that it is not none.
        for key, value in self.component.surfaces.items():
            self.assertTrue(value.zmx_surf is not None)





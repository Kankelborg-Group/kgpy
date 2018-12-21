
from abc import ABC, abstractmethod
from unittest import TestCase

from . import Zemax
from.zemax.zemax import TestZemax
from .component import DefaultPrimary


class System(ABC):
    """
    The System class simulates an entire optical system, and is represented as a series of Components
    """

    def __init__(self, zmx_path):
        """
        This constructor initializes all Components defined in the abstract property component list.
        :param zmx_path: Full path to Zemax design file
        """

        # Save path to variable
        self.zmx_path = zmx_path

        # Open Zemax object so we can read in the surfaces
        self.zmx = Zemax(zmx_path)

        # Loop through component dictionary, initialize each object, and place in dictionary for later access
        self.components = {}
        for component in self.component_list:
            c = component(self.zmx)
            self.components[c.name] = c

    @property
    @abstractmethod
    def name(self):
        """
        Name of this system
        :return: Human-readable name of this system
        :rtype: str
        """
        return 'default system name'

    @property
    @abstractmethod
    def component_list(self):
        """
        To define a concrete System, we need to define a list of Component objects we would like to initialize
        :return: list of uninstantiated objects inheriting from Component
        """

        return [DefaultPrimary]


class DefaultSystem(System):
    """
    Default optical system to test the abstract System class
    """

    @property
    def name(self):
        """
        :return: Default name defined in abstract superclass
        :rtype: str
        """
        return super().name

    @property
    def component_list(self):
        """
        :return: Default component list defined in abstract superclass
        """
        return super().component_list


class TestSystem(TestCase):
    """
    Test class for exercising the System class. Uses the DefaultSystem class as a concrete example
    """

    def setUp(self):

        pass

    def test__init__(self):
        """
        Test the constructor by checking that there are no undefined components in the component dictionary
        :return: None
        """

        # Initialize test Zemax design
        sys = DefaultSystem(TestZemax.test_path)

        # Check that none of the components are none
        for key, value in sys.components.items():
            self.assertTrue(value is not None)


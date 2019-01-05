

from unittest import TestCase


__all__ = ['System']


class System:
    """
    The System class simulates an entire optical system, and is represented as a series of Components.
    This class is intended to be a drop-in replacement for a Zemax system.
    """

    def __init__(self, name, components):
        """
        This constructor initializes all Components defined in the abstract property component list.
        :param name: Human-readable name of the system
        :param components: List of initial components for the system
        :type name: str
        :type components: list[kgpy.optics.Component]
        """

        # Save input arguments to class variables
        self.name = name
        self.components = components




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


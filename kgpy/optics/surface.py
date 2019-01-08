
from unittest import TestCase

__all__ = ['Surface']


class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors
    """

    def __init__(self, name, thickness=0.0, comment=''):
        """
        Constructor for the Surface class. Currently only saves input arguments
        :param name: Human-readable name of the surface
        :param thickness: Thickness of the surface along the current z-direction, measured in mm
        :param comment: Additional description of this surface
        :type name: str
        :type thickness: float
        :type comment: str
        """

        self.name = name
        self.thickness = thickness
        self.comment = comment

        # Assume that global z coordinate is zero initially. Z-coordinate is set by placing Surface into Component.
        self.X =

    @property
    def z(self):
        """
        :return: Global z-coordinate of surface
        :rtype: float
        """
        return self._z

    @z.setter
    def z(self, z):
        """
        :param z: New global z-coordinate of surface
        :type z: float
        :return: None
        """
        self._z = z




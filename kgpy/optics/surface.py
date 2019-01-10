
from unittest import TestCase
import numpy as np
import quaternion

from kgpy.math import CoordinateSystem,Vector

__all__ = ['Surface']


class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors
    """

    def __init__(self, name: str, thickness=0.0, comment=''):
        """
        Constructor for the Surface class. Currently only saves input arguments
        :param name: Human-readable name of the surface
        :param thickness: Thickness of the surface along the current z-direction, measured in mm
        :param comment: Additional description of this surface
        :type thickness: float
        :type comment: str
        """

        # Save input arguments as class variables
        self.name = name
        self.thickness = thickness
        self.comment = comment

        # Initialize the coordinate system to the global coordinate system
        self.cs = CoordinateSystem([0, 0, 0], [0, 0, 0, 0])

    @property
    def T(self):
        """
        Thickness vector
        :return: Vector pointing from front face of surface to back face of surface
        """
        return self.thickness * self.cs.zh







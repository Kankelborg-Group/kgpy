
from unittest import TestCase
import numpy as np
import quaternion


from kgpy.math import CoordinateSystem, Vector, GlobalCoordinateSystem

__all__ = ['Surface']


class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    def __init__(self, name: str, thickness: float = 0.0, comment: str = ''):
        """
        Constructor for the Surface class.
        This constructor places the surface at the origin of the global coordinate system, it needs to be moved into
        place after the call to this function.
        :param name: Human-readable name of the surface
        :param thickness: Thickness of the surface along the local z-direction, measured in mm
        :param comment: Additional description of this surface
        """

        # Save input arguments as class variables
        self.name = name
        self.thickness = thickness
        self.comment = comment

        # Initialize the coordinate system to the global coordinate system
        self.cs = GlobalCoordinateSystem()

    @property
    def T(self):
        """
        Thickness vector
        :return: Vector pointing from front face of surface to back face of surface
        """
        return self.thickness * self.cs.zh







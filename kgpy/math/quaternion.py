
import numpy as np

__all__ = ['Quaternion', 'QuaternionSystem']


class Quaternion:
    """
    Represents a 4D quaternion.
    """

    def __init__(self, w, r):
        """
        Construct a new quaternion from its scalar and vector components
        :param w: Scalar component of quaternion
        :param r: Vector component of quaternion
        :type w: float
        :type r: kgpy.math.Vector
        """

        # Save input arguments as class variables
        self.w = w
        self.r = r


class QuaternionSystem:
    """
    A QuaternionSystem is similar to a VectorSystem, and is composed of a Quaternion and a CoordinateSystem.
    """

    def __init__(self, Q, cs):
        """
        Construct a new QuaternionSystem using a quaternion and coordinate system
        :param Q: Quaternion defined in the CoordinateSystem cs
        :param cs: Coordinate system
        :type Q: kgpy.math.Quaternion
        :type cs: kgpy.math.CoordinateSystem
        """

        # Save input arguments
        self.Q = Q
        self.cs = cs

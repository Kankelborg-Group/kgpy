
import numpy as np
from . import Vector

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
        :type r: kgpy.math.Vector or list[float]
        """

        # Convert to vector if list is provided as an argument
        if isinstance(r, list):
            r = Vector(r)

        # Save input arguments as class variables
        self.w = w
        self.r = r

    @property
    def x(self):
        return self.r.x

    @x.setter
    def x(self, x):
        self.r.x = x

    @property
    def y(self):
        return self.r.y

    @y.setter
    def y(self, y):
        self.r.y = y

    @property
    def z(self):
        return self.r.z

    @z.setter
    def z(self, z):
        self.r.z = z

    def __mul__(self, other):
        """
        :param other: Entity to multiply this quaternion by
        :type other: kgpy.math.Quaternion
        :return: Product of this quaternion with other
        :rtype: kgpy.math.Quaternion
        """

        # Quaternion product (Hamilton product)
        if isinstance(other, Quaternion):

            # Value of each quaternion component
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w

            return Quaternion(w, [x, y, z])

        # Undefined product
        else:
            return None


class QuaternionSystem:
    """
    A QuaternionSystem is similar to a VectorSystem, and is composed of a Quaternion and a CoordinateSystem.
    """

    def __init__(self, Q, cs):
        """
        Construct a new QuaternionSystem using a quaternion and coordinate system
        :param Q: Quaternion defined in the CoordinateSystem cs
        :param cs: Coordinate system
        :type Q: quaternion.quaternion
        :type cs: kgpy.math.CoordinateSystem
        """

        # Save input arguments
        self.Q = Q
        self.cs = cs


from typing import List, Union
from copy import deepcopy
import astropy.units as u
import quaternion
from . import Vector

__all__ = ['CoordinateSystem', 'GlobalCoordinateSystem']


class CoordinateSystem:
    """
    A coordinate system is described by a 3D translation and a 3D rotation from some global coordinate system.
    """

    # Unit vectors for the global coordinate system (base unit vectors)
    xh_g = Vector([1, 0, 0])
    yh_g = Vector([0, 1, 0])
    zh_g = Vector([0, 0, 1])

    def __init__(self, X: Union[List[float], u.Quantity, Vector], Q: Union[List[float], quaternion.quaternion]):
        """
        Defines a new coordinate system using a vector (translation) and a quaternion (rotation)
        :param X: Vector pointing from origin of global coordinate system to the origin of this coordinate system
        :param Q: Quaternion representing the 3D rotation of this coordinate system with respect to the global
        coordinate system.
        """

        # Convert X to vector if it isn't already
        if isinstance(X, list):
            X = Vector(X)
        elif isinstance(X, u.Quantity):
            X = Vector(X)

        # Convert Q to quaternion if it isn't already
        if isinstance(Q, list):
            Q = quaternion.quaternion(*Q)

        # Save input arguments to class variables
        self.X = X
        self.Q = Q

    @property
    def xh(self) -> Vector:
        """
        :return: x-hat unit vector for this coordinate system
        """
        xh = quaternion.rotate_vectors(self.Q, self.xh_g)
        return Vector(xh)

    @property
    def yh(self) -> Vector:
        """
        :return: y-hat unit vector for this coordinate system
        """
        yh = quaternion.rotate_vectors(self.Q, self.yh_g)
        return Vector(yh)

    @property
    def zh(self) -> Vector:
        """
        :return: z-hat unit vector for this coordinate system
        """
        zh = quaternion.rotate_vectors(self.Q, self.zh_g)
        return Vector(zh)

    def __str__(self) -> str:
        """
        :return: Human-readable string representation of this coordinate system
        """
        return self.X.__str__() + ', ' +  self.Q.__str__()

    def __add__(self, other: Vector) -> 'CoordinateSystem':
        """
        Add a vector to the current coordinate system.
        This operation translates the origin of the coordinate system by the Vector other.
        :param other: Vector to translate the coordinate system by
        :return: New coordinate system after translation
        """

        # Check if the right operand is an instance of Vector
        if isinstance(other, Vector):

            # Make sure to deepcopy the quaternion object to decouple the new coordinate system
            return CoordinateSystem(self.X + other, deepcopy(self.Q))

        else:
            raise ValueError('Only Vectors can be added to a coordinate system')

    def __eq__(self, other: 'CoordinateSystem') -> bool:
        """
        Check if two coordinate systems are equal by checking if the translation vectors are equal and the rotation
        quaternions are equal.
        :param other: Another coordinate system to compare
        :return: True if the two coordinate systems are the same, False if not.
        """

        # Check if the translation and rotation is the same in the other coordinate system
        a = self.X == other.X
        b = self.Q == other.Q

        # return True if the translation and rotation is the same for both coordinate systems
        return a and b


class GlobalCoordinateSystem(CoordinateSystem):
    """
    The global coordinate system is a coordinate system with no translations or rotations
    """

    def __init__(self):

        super().__init__([0, 0, 0] * u.mm, quaternion.from_euler_angles(0, 0, 0))



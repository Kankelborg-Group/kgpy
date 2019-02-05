
from typing import List, Union
from copy import deepcopy
import astropy.units as u
import quaternion as q
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

    def __init__(self, X: Union[List[float], u.Quantity, Vector], Q: Union[List[float], q.quaternion],
                 translation_first = True):
        """
        Defines a new coordinate system using a vector (translation) and a quaternion (rotation).
        Rotations and translations are performed using the current orientation of the coordinate system, this means
        that the order of the rotation and translation matters.
        :param X: Vector pointing from origin of global coordinate system to the origin of this coordinate system
        :param Q: Quaternion representing the 3D rotation of this coordinate system with respect to the global
        coordinate system.
        :param translation_first: flag to indicate whether the translation operation should occur before the rotation
        operation.
        """

        # Convert X to vector if it isn't already
        if isinstance(X, list):
            X = Vector(X)
        elif isinstance(X, u.Quantity):
            X = Vector(X)

        # Convert Q to quaternion if it isn't already
        if isinstance(Q, list):
            Q = q.quaternion(*Q)

        # Save input arguments to class variables
        self.X = X
        self.Q = Q

        # If the translation operation occurs first, we do not have to rotate the translation Vector.
        # Otherwise, if the rotation operation occurs first, we do have to rotate the translation Vector.
        if not translation_first:
            self.X = q.rotate_vectors(self.Q, self.X)

    @property
    def xh(self) -> Vector:
        """
        :return: x-hat unit vector for this coordinate system
        """
        xh = q.rotate_vectors(self.Q, self.xh_g)
        return Vector(xh)

    @property
    def yh(self) -> Vector:
        """
        :return: y-hat unit vector for this coordinate system
        """
        yh = q.rotate_vectors(self.Q, self.yh_g)
        return Vector(yh)

    @property
    def zh(self) -> Vector:
        """
        :return: z-hat unit vector for this coordinate system
        """
        zh = q.rotate_vectors(self.Q, self.zh_g)
        return Vector(zh)

    def __str__(self) -> str:
        """
        :return: Human-readable string representation of this coordinate system
        """
        return self.X.__str__() + ', ' +  self.Q.__str__()

    def __eq__(self, other: 'CoordinateSystem') -> bool:
        """
        Check if two coordinate systems are equal by checking if the translation vectors are equal and the rotation
        quaternions are equal.
        :param other: Another coordinate system to compare
        :return: True if the two coordinate systems are the same, False if not.
        """

        # Check that the right operand is an instance of a coordinate system
        if isinstance(other, CoordinateSystem):

            # Check if the translation and rotation is the same in the other coordinate system
            a = self.X == other.X
            b = self.Q == other.Q

            # return True if the translation and rotation is the same for both coordinate systems
            return a and b

        else:
            return NotImplemented

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
            return NotImplemented

    # Define the reverse addition operator so that both Vector + CoordinateSystem and CoordinateSystem + Vector are
    # allowed.
    __radd__ = __add__

    def __mul__(self, other: q.quaternion) -> 'CoordinateSystem':
        """
        Multiply the coordinate system by the quaternion other.
        This operation rotates the coordinate system by the quaternion other.
        :param other: Quaternion to rotate the coordinate system by
        :return: The rotated coordinate system
        """

        # Check if the right operand is an instance of a q.quaternion
        if isinstance(other, q.quaternion):

            # As in the __add__ function, we deepcopy the Vector object to decouple the new coordinate system
            return CoordinateSystem(deepcopy(self.X), self.Q * other)

        else:
            return NotImplemented

    # Note that the reverse multiplication operation is not defined here since quaternion products do not commute

    def __matmul__(self, other: 'CoordinateSystem') -> 'CoordinateSystem':
        """
        Compute the composition of two coordinate systems.
        We define this composition
        We define this composition by adding the two translation vectors from each coordinate system and multiplying
        the two rotation quaternions.
        This function can be interpreted as translating/rotating other by self.
        :param other: The other coordinate system to be transformed
        :return: A new coordinate system representing the composition of self and other.
        """

        X = self.X + q.rotate_vectors(self.Q, other.X)
        Q = self.Q * other.Q

        # Return a coordinate system with translation before rotation because we don't want an additional modification
        # of the translation vector.
        return CoordinateSystem(X, Q)


class GlobalCoordinateSystem(CoordinateSystem):
    """
    The global coordinate system is a coordinate system with no translations or rotations
    """

    def __init__(self):

        super().__init__([0, 0, 0] * u.mm, q.from_euler_angles(0, 0, 0))




from typing import List, Union
from copy import deepcopy
import numpy as np
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
                 translation_first: bool = True):
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
        self.X = X      # type: Vector
        self.Q = Q      # type: q.quaternion
        self.translation_first = translation_first

        # If the translation operation occurs first, we do not have to rotate the translation Vector.
        # Otherwise, if the rotation operation occurs first, we do have to rotate the translation Vector.
        if not translation_first:
            self.X = self.X.rotate(self.Q)

    @property
    def xh(self) -> Vector:
        """
        :return: x-hat unit vector for this coordinate system
        """
        return self.xh_g.rotate(self.Q)

    @property
    def yh(self) -> Vector:
        """
        :return: y-hat unit vector for this coordinate system
        """
        return self.yh_g.rotate(self.Q)

    @property
    def zh(self) -> Vector:
        """
        :return: z-hat unit vector for this coordinate system
        """
        return self.zh_g.rotate(self.Q)

    def __str__(self) -> str:
        """
        :return: Human-readable string representation of this coordinate system
        """
        return 'CoordinateSystem(' + self.X.__str__() + ', ' + self.Q.__str__() + ')'

    __repr__ = __str__

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
        This function can be interpreted as translating/rotating other by self.
        :param other: The other coordinate system to be transformed
        :return: A new coordinate system representing the composition of self and other.
        """

        # Compute new system attributes
        X = self.X + other.X.rotate(self.Q)
        Q = self.Q * other.Q

        # Return a coordinate system with translation before rotation because we don't want an additional modification
        # of the translation vector.
        return CoordinateSystem(X, Q)

    @property
    def inverse(self):
        """
        Compute the inverse of this coordinate system.
        The inverse is defined as the coordinate system that when composed with this coordinate system, equals the
        global coordinate system (no rotations / no translations).
        :return: The inverse of this coordinate system.
        """

        # Invert attributes
        X = -self.X
        Q = self.Q.conj()

        return CoordinateSystem(X, Q, translation_first=False)

    def diff(self, other: 'CoordinateSystem'):
        """
        Calculate the difference between this coordinate system and another coordinate system.
        In other words: what coordinate system would need to be composed with other to result in this coordinate system.
        :param other: The other coordinate system to remove from this coordinate system
        :return: New coordinate system representing the difference between the two coordinate systems.
        """

        return self @ other.inverse

    def isclose(self, other: 'CoordinateSystem') -> bool:
        """
        Check if CoordinateSystem other is nearly equal to self.
        This function checks if both the translation vector and the rotation quaternion in both systems are equal.
        :param other: Another CoordinateSystem to compare to
        :return: True if the CoordinateSystems are nearly equal, false otherwise
        """

        # Compare the vectors
        a = self.X.isclose(other.X)

        # Compare the quaternions
        b = np.isclose(self.Q, other.Q).all()

        return a and b

    def xy_intercept(self, v1: Vector, v2: Vector) -> Union[Vector, None]:
        """
        Determine the point where the line formed by x1 and x2 intercepts the x-y plane of this coordinate system.
        This function is based off of the code provided in this answer: https://stackoverflow.com/a/18543221
        :param v1: Start-point of the line
        :param v2: End-point of the lien
        :return: Vector pointing from the origin of this coordinate system to the point of intersection, or None if an
        intersection cannot be found.
        """

        # Change parameter names to be the same as the stackoverflow answer cited above
        epsilon=1e-6 * v1.X.unit
        p0 = v1
        p1 = v2
        p_co = self.X
        p_no = self.zh

        # Compute vector from start to end point
        u = p1 - p0

        #
        dot = p_no.dot(u)

        if abs(dot) > epsilon:

            w = p0 - p_co
            fac = -p_no.dot(w) / dot
            if (fac >= 0.0) and (fac <= 1.0):

                u = fac * u

                return p0 + u

        return None


class GlobalCoordinateSystem(CoordinateSystem):
    """
    The global coordinate system is a coordinate system with no translations or rotations
    """

    def __init__(self):

        super().__init__([0, 0, 0] * u.mm, q.from_euler_angles(0, 0, 0))



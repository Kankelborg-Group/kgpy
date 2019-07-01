
import enum
import typing as tp
import numpy as np
import astropy.units as u
import quaternion as q

from kgpy import math

__all__ = ['Vector']


class Axis(enum.IntEnum):
    x = enum.auto()
    y = enum.auto()
    z = enum.auto()


class Vector:
    """
    Represents a 3D vector
    """

    def __init__(self):

        # Save input to class variable
        self.X = [0, 0, 0] * u.dimensionless_unscaled       # type: u.Quantity

    @property
    def x(self) -> u.Quantity:
        return self.X[Axis.x]

    @x.setter
    def x(self, x: u.Quantity):
        self.X[Axis.x] = x

    @property
    def y(self) -> u.Quantity:
        return self.X[Axis.y]

    @y.setter
    def y(self, y: u.Quantity):
        self.X[Axis.y] = y

    @property
    def z(self) -> u.Quantity:
        return self.X[Axis.z]

    @z.setter
    def z(self, z: u.Quantity):
        self.X[Axis.z] = z

    def __eq__(self, other: 'Vector'):

        return np.all(self.X == other.X.to(self.X.unit))

    def __neg__(self) -> 'Vector':
        """
        Reverse the direction of a vector
        :return: a new vector with the values of each component negated
        """
        v = self.copy()

        v.X = -v.X

        return v

    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Add two vectors together
        :param other: Another vector
        :return: The sum of both vectors
        """
        v = self.copy()
        v.X += other.X
        return v

    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Subtract other Vector from this Vector
        :param other: Another vector
        :return: The difference of this Vector and the Vector other
        """
        return self.__add__(-other)

    def __mul__(self, other: tp.Union[int, float, u.Quantity, u.Unit]) -> 'Vector':
        """
        Multiplication of a scalar and a vector
        :param other: a scalar value
        :return: The original vector where every component has been scaled by other
        """
        v = self.copy()
        v.X *= other
        return v

    # This is needed to make astropy.units respect our version of the __rmul__ op, for more information see
    # https://stackoverflow.com/a/41948659
    __array_priority__ = 10000000000

    # Make the reverse multiplication operation the same as the multiplication operation
    __rmul__ = __mul__

    def dot(self, other: 'Vector') -> u.Quantity:
        """
        Execute the dot product of this Vector with another Vector
        :param other: The other Vector to dot this Vector into
        :return: Result of the dot product
        """

        return self.X.dot(other.X)

    def cross(self, other: 'Vector') -> 'Vector':
        """
        Calculate the cross product of this Vector with another Vector
        :param other: Another vector to cross into this Vector
        :return: a Vector orthogonal to both of the input Vectors.
        """

        v = self.copy()
        v.X = np.cross(self.X, other.X.to(self.X.unit)) << (self.X.unit**2)

        return v

    def rotate(self, Q: math.geometry.Quaternion):

        v = self.copy()

        # quaternion.rotate_vectors operates on numpy.ndarray surf_types (not astropy.units.Quantity surf_types), so we need to
        # explicitly include the unit
        v.X = q.rotate_vectors(Q, self.X) << self.X.unit

        return v


    @property
    def mag(self):
        """
        Compute the magnitude of the Vector
        :return: L2 norm of the vector
        """

        return np.linalg.norm(self.X) << self.X.unit

    def normalize(self) -> 'Vector':

        return self * (1 / self.mag)

    def angle_between(self, other: 'Vector'):

        a = self.cross(other).mag

        b = self.mag
        c = other.mag

        return np.arcsin(a / (b * c)) << u.rad

    def isclose(self, other: 'Vector'):
        """
        Check if two Vectors have the same components to within some tolerance
        :param other: Another vector to compare against
        :return: True if the two vectors are nearly the same, False otherwise.
        """

        # Take the difference of the two values so that we can compare to zero. This lets us use the np.isclose()
        # function and not care about the units of self or other (as long as they're compatible of course).
        x = self - other

        return np.isclose(x.X.value, [0, 0, 0]).all()

    def copy(self) -> 'Vector':

        c = type(self)()

        c.X = self.X.copy()

        return c

    def __str__(self) -> str:
        """
        Print a string representation of the vector
        :return: The string representation of the underlying numpy.ndarray
        """
        return 'vector(' + str(self.X) + ')'

    def __repr__(self) -> str:

        return self.__str__()

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Interpret the vector as a numpy.ndarray
        :param dtype: Type of the output
        :return:
        """
        try:
            return np.array(self.X.to_value(u.dimensionless_unscaled), dtype=dtype)
        except (u.UnitsError, TypeError):
            raise TypeError('only dimensionless scalar quantities can be '
                            'converted to numpy arrays')









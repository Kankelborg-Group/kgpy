
import enum
import typing as tp
import numpy as np
import astropy.units as u
import quaternion as q

from kgpy import math

__all__ = ['Vector']


class Axis(enum.IntEnum):
    x = 0
    y = 1
    z = 2


class Vector:
    """
    Represents a 3D vector
    """

    def __init__(self, components: tp.Optional[u.Quantity] = None):

        if components is None:
            components = [0, 0, 0] * u.dimensionless_unscaled

        # Save input to class variable
        self._components = components

    @property
    def x(self) -> u.Quantity:
        return self._components[Axis.x]

    @property
    def y(self) -> u.Quantity:
        return self._components[Axis.y]

    @property
    def z(self) -> u.Quantity:
        return self._components[Axis.z]

    @property
    def components(self) -> u.Quantity:
        return self._components

    def __eq__(self, other: 'Vector'):

        return np.all(self._components == other._components.to(self._components.unit))

    def __neg__(self) -> 'Vector':
        """
        Reverse the direction of a vector
        :return: a new vector with the values of each component negated
        """

        components = self._components.__neg__()
        return type(self)(components)

    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Add two vectors together
        :param other: Another vector
        :return: The sum of both vectors
        """
        components = self._components.__add__(other._components)
        return type(self)(components)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Subtract other Vector from this Vector
        :param other: Another vector
        :return: The difference of this Vector and the Vector other
        """
        return self.__add__(other.__neg__())

    def __mul__(self, other: tp.Union[int, float, u.Quantity, u.Unit]) -> 'Vector':
        """
        Multiplication of a scalar and a vector
        :param other: a scalar value
        :return: The original vector where every component has been scaled by other
        """
        components = self._components.__mul__(other)
        return type(self)(components)

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

        return self._components.dot(other._components)

    def cross(self, other: 'Vector') -> 'Vector':
        """
        Calculate the cross product of this Vector with another Vector
        :param other: Another vector to cross into this Vector
        :return: a Vector orthogonal to both of the input Vectors.
        """

        components = np.cross(self._components, other._components) << (self._components.unit * other._components.unit)

        return type(self)(components)

    def rotate(self, value: 'Vector', inverse=False):

        print(value)

        a = value.x
        b = value.y
        c = value.z

        r_x = np.array([
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a), np.cos(a)]
        ])

        r_y = np.array([
            [np.cos(b), 0, np.sin(b)],
            [0, 1, 0],
            [-np.sin(b), 0, np.cos(b)],
        ])

        r_z = np.array([
            [np.cos(c), -np.sin(c), 0],
            [np.sin(c), np.cos(c), 0],
            [0, 0, 1]
        ])

        if not inverse:
            r = r_z @ r_y @ r_x

        else:
            r = r_x @ r_y @ r_z

        components = r @ self.components

        return type(self)(components)


    @property
    def mag(self):
        """
        Compute the magnitude of the Vector
        :return: L2 norm of the vector
        """

        return np.linalg.norm(self.components.value) << self.components.unit

    @property
    def normalized(self) -> 'Vector':

        return self * (1 / self.mag)

    def angle_between(self, other: 'Vector') -> u.Quantity:

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

        return np.isclose(x._components.value, [0, 0, 0]).all()

    def __str__(self) -> str:
        """
        Print a string representation of the vector
        :return: The string representation of the underlying numpy.ndarray
        """
        return 'vector(' + str(self._components) + ')'

    def __repr__(self) -> str:

        return self.__str__()

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Interpret the vector as a numpy.ndarray
        :param dtype: Type of the output
        :return:
        """
        try:
            return np.array(self._components.to_value(u.dimensionless_unscaled), dtype=dtype)
        except (u.UnitsError, TypeError):
            raise TypeError('only dimensionless scalar quantities can be '
                            'converted to numpy arrays')









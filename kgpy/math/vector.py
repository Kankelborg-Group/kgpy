
from typing import Union, List
from numbers import Real
import numpy as np
import astropy.units as u
import quaternion as q

__all__ = ['Vector']


class Vector:
    """
    Represents a 3D vector
    """

    def __init__(self, X: Union[List[Real], np.ndarray, u.Quantity]):
        """
        Construct a new vector from a list, array or Quantity, checking for correct shape and size.
        :param X: 1D list, array or Quantity with only three elements
        """

        if isinstance(X, list):
            X = X * u.dimensionless_unscaled

        # Check that the input is the correct type
        if isinstance(X, np.ndarray):
            X = X * u.dimensionless_unscaled

        if not isinstance(X, u.Quantity):
            raise TypeError('Input array X must be List[float], numpy.ndarray, Quantity')

        # Check that the input is 1D
        if len(X.shape) != 1:
            raise TypeError('Incorrect shape for input array X')

        # Check that the input has only three elements
        if X.shape[0] != 3:
            raise TypeError('Input array X does not have three elements')

        # Save input to class variable
        self.X = X  # type: u.Quantity

    @property
    def x(self) -> u.Quantity:
        return self.X[0]

    @x.setter
    def x(self, x: u.Quantity):
        self.X[0] = x

    @property
    def y(self) -> u.Quantity:
        return self.X[1]

    @y.setter
    def y(self, y: u.Quantity):
        self.X[1] = y

    @property
    def z(self) -> u.Quantity:
        return self.X[2]

    @z.setter
    def z(self, z: u.Quantity):
        self.X[2] = z

    def __eq__(self, other: Union[Real, u.Quantity, 'Vector']):
        """
        Test if two Vectors are the same or test if every component of this vector is equal to the same scalar value.
        Two vectors are the same if all their elements are the same.
        :param other: Another Vector to compare to this vector, or a scalar
        :return: True if all elements are equal, false otherwise.
        """
        if isinstance(other, self.__class__):
            return np.all(self.X == other.X)
        elif isinstance(other, Real) or isinstance(other, u.Quantity):
            return np.all(self.X == other)
        else:
            raise TypeError('right argument is not a Vector, Quantity or Real type')

    def __neg__(self) -> 'Vector':
        """
        Reverse the direction of a vector
        :return: a new vector with the values of each component negated
        """

        return Vector(-self.X)

    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Add two vectors together
        :param other: Another vector
        :return: The sum of both vectors
        """
        if isinstance(other, Vector):
            X = self.X + other.X
            return Vector(X)
        else:
            return NotImplemented

    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Subtract other Vector from this Vector
        :param other: Another vector
        :return: The difference of this Vector and the Vector other
        """
        if isinstance(other, Vector):
            return self.__add__(-other)
        else:
            return NotImplemented

    def __mul__(self, other: Union[Real, u.Quantity, u.Unit]) -> 'Vector':
        """
        Multiplication of a scalar and a vector
        :param other: a scalar value
        :return: The original vector where every component has been scaled by other
        """
        if isinstance(other, (Real, u.Quantity, u.Unit)):
            X = other * self.X
            return Vector(X)
        else:
            return NotImplemented

    # This is needed to make astropy.units respect our version of the __rmul__ op, for more information see
    # https://stackoverflow.com/a/41948659
    __array_priority__ = 10000000000

    # Make the reverse multiplication operation the same as the multiplication operation
    __rmul__ = __mul__

    def dot(self, other: 'Vector') -> Real:
        """
        Execute the dot product of this Vector with another Vector
        :param other: The other Vector to dot this Vector into
        :return: Result of the dot product
        """

        # Use numpy.ndarray.dot() to calculate the dot product
        return self.X.dot(other.X)

    def cross(self, other: 'Vector') -> 'Vector':
        """
        Calculate the cross product of this Vector with another Vector
        :param other: Another vector to cross into this Vector
        :return: a Vector orthogonal to both of the input Vectors.
        """

        # Use numpy.ndarray.cross() to calculate the cross product
        return Vector(np.cross(self.X, other.X) * self.X.unit * other.X.unit)

    def rotate(self, Q: q.quaternion):
        """
        Rotate self by the quaternion Q
        :param Q: Quaternion describing the rotation of the vector
        :return: Rotated vector
        """

        # quaternion.rotate_vectors operates on numpy.ndarray surf_types (not astropy.units.Quantity surf_types), so we need to
        # explicitly include the unit
        x = q.rotate_vectors(Q, self.X) * self.X.unit

        return Vector(x)

    @property
    def mag(self):
        """
        Compute the magnitude of the Vector
        :return: L2 norm of the vector
        """

        return np.linalg.norm(self.X) * self.X.unit

    def normalize(self) -> 'Vector':

        return self * (1 / self.mag)

    def angle_between(self, other: 'Vector'):

        a = self.cross(other).mag

        b = self.mag
        c = other.mag

        return np.arcsin(a / (b * c))

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









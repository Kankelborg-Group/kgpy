
from unittest import TestCase
from typing import Union, List
import numpy as np
import astropy.units as u

__all__ = ['Vector']


class Vector:
    """
    Represents a 3D vector
    """

    def __init__(self, X: Union[List[float], np.ndarray, u.Quantity]):
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

    #
    # def __array__(self, dtype=None):
    #     if dtype:
    #         return self.X.astype(dtype)
    #     else:
    #         return self.X


    @property
    def x(self):
        return self.X[0]

    @x.setter
    def x(self, x):
        self.X[0] = x

    @property
    def y(self):
        return self.X[1]

    @y.setter
    def y(self, y):
        self.X[1] = y

    @property
    def z(self):
        return self.X[2]

    @z.setter
    def z(self, z):
        self.X[2] = z

    def __eq__(self, other: 'Vector'):
        """
        Test if two Vectors are the same.
        Two vectors are the same if all their elements are the same.
        :param other: Another Vector to compare to this vector
        :return: True if all elements are equal, false otherwise.
        """
        return np.all(self.X == other.X)

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
            raise TypeError

    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Add two vectors together
        :param other: Another vector
        :return: The sum of both vectors
        """
        if isinstance(other, Vector):
            X = self.X - other.X
            return Vector(X)
        else:
            raise TypeError

    def __mul__(self, other: u.Quantity) -> 'Vector':
        """
        Multiplication of a scalar and a vector
        :param other: A scalar value
        :return: The original vector where every component has been scaled by other
        """
        if isinstance(other, u.Quantity):
            X = other * self.X
            return Vector(X)
        else:
            raise TypeError

    # This is needed to make astropy.units respect our version of the __rmul__ op, for more information see
    # https://stackoverflow.com/a/41948659
    __array_priority__ = 10000000

    # Make the reverse multiplication operation the same as the multiplication operation
    __rmul__ = __mul__

    def __str__(self) -> str:
        """
        Print a string representation of the vector
        :return: The string representation of the underlying numpy.ndarray
        """
        return 'vector(' + str(self.X) + ')'

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










from unittest import TestCase
from typing import Union
import numpy as np

__all__ = ['Vector']


class Vector:
    """
    Represents a 3D vector
    """

    def __init__(self, X):
        """
        Construct a new vector from an array, checking for correct shape and size.
        :param X: 1D array with only three elements
        :type X: numpy.ndarray or list[float]
        """

        if isinstance(X, list):
            X = np.array(X)

        # Check that the input is the correct type
        if not isinstance(X, np.ndarray):
            raise TypeError('Incorrect type for input array X')

        # Check that the input is 1D
        if len(X.shape) != 1:
            raise TypeError('Incorrect shape for input array X')

        # Check that the input has only three elements
        if X.shape[0] != 3:
            raise TypeError('Input array X does not have three elements')

        # Save input to class variable
        self.X = X

    def __array__(self, dtype=None):
        if dtype:
            return self.X.astype(dtype)
        else:
            return self.X


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
            X = np.add(self, other)
            return Vector(X)
        else:
            raise TypeError

    def __mul__(self, other: float) -> 'Vector':
        """
        Multiplication of a scalar and a vector
        :param other: A scalar value
        :return: The original vector where every component has been scaled by other
        """
        if isinstance(other, float):
            X = other * self.X
            return Vector(X)
        else:
            raise TypeError

    # Reverse multiplication should behave in the same way as forward multiplication
    __rmul__ = __mul__

    def __str__(self) -> str:
        """
        Print a string representation of the vector
        :return: The string representation of the underlying numpy.ndarray
        """
        return self.X.__str__()


class TestVector(TestCase):

    def test__eq__(self):

        # Declare two different vectors
        v0 = Vector([1, 1, 1])
        v1 = Vector([1, 1, 0])

        # Assert vector is equal to itself
        self.assertTrue(v0 == v0)

        # Assert two different vectors are not equal
        self.assertFalse(v0 == v1)

    def test__add__(self):

        v0 = Vector([1, 0, 0])
        v1 = Vector([0, 1, 0])
        v2 = Vector([1, 1, 0])

        v = v0 + v1

        self.assertTrue(v == v2)



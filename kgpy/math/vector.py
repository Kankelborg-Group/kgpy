
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

    def __mul__(self, other):
        if isinstance(other, float):
            self.X = other * self.X







import numpy as np

__all__ = ['Vector', 'VectorSystem']


class Vector:
    """
    Represents a 3D vector
    """

    def __init__(self, X):
        """
        Construct a new vector from an array, checking for correct shape and size.
        :param X: 1D array with only three elements
        :type X: numpy.ndarray
        """

        # Check that the input is the correct type
        if type(X) != np.ndarray:
            raise TypeError('Incorrect type for input array X')

        # Check that the input is 1D
        if len(X.shape) != 1:
            raise TypeError('Incorrect shape for input array X')

        # Check that the input has only three elements
        if X.shape != 3:
            raise TypeError('Input array X does not have three elements')

        # Save input to class variable
        self.X = X

    @property
    def x(self):
        return self.X[0]

    @property
    def y(self):
        return self.X[1]

    @property
    def z(self):
        return self.X[2]


class VectorSystem:
    """
    A VectorSystem is an object comprised of a vector and a coordinate system.
    It represents a vector in a particular coordinate system
    """

    def __init__(self, X, cs):
        """
        Construct ray from coordinate system and vector
        :param X: Vector defined in the coordinate system cs
        :param cs: Coordinate system of the Ray
        :type X: kgpy.math.Vector
        :type cs: kgpy.math.CoordinateSystem
        """

        # Save input arguments as class variables
        self.X = X
        self.cs = cs

    def rotate(self, Q):
        """
        Rotate the Vector self.X by the Quaternion Q.Q. Note that the coordinate systems of self and Q must be the same
        for this operation to be defined.
        :param Q: QuaternionSystem to rotate about.
        :type Q: kgpy.math.QuaternionSystem
        :return: kgpy.math.VectorSystem
        """
        pass



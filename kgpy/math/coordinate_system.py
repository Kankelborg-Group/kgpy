
from unittest import TestCase
from typing import List, Union
import numpy as np
import quaternion
from . import Vector


class CoordinateSystem:
    """
    A coordinate system is described by a 3D translation and a 3D rotation from some global coordinate system.
    """

    xh_g = Vector([1, 0, 0])
    yh_g = Vector([0, 1, 0])
    zh_g = Vector([0, 0, 1])

    def __init__(self, X: Union[List[float], Vector], Q: Union[List[float], quaternion.quaternion]):
        """
        Defines a new coordinate system using a vector (translation) and a quaternion (rotation)
        :param X: Vector pointing from origin of global coordinate system to the origin of this coordinate system
        :param Q: Quaternion representing the 3D rotation of this coordinate system with respect to the global
        coordinate system.
        """

        # Convert X to vector if it isn't already
        if isinstance(X, list):
            X = Vector(X)

        # Convert Q to quaternion if it isn't already
        if isinstance(Q, list):
            Q = quaternion.quaternion(*Q)

        # Save input arguments to class variables
        self.X = X
        self.Q = Q

    @property
    def xh(self):
        xh = quaternion.rotate_vectors(self.Q, self.xh_g.X)
        return Vector(xh)

    @property
    def yh(self):
        yh = quaternion.rotate_vectors(self.Q, self.yh_g.X)
        return Vector(yh)

    @property
    def zh(self):
        zh = quaternion.rotate_vectors(self.Q, self.zh_g.X)
        return Vector(zh)


class TestCoordinateSystem(TestCase):

    def setUp(self):

        X = Vector([0, 0, 0])
        Q = quaternion.from_euler_angles(np.pi/2, 0, 0)
        print(Q)
        self.cs = CoordinateSystem(X, Q)

    def test_xh(self):

        print(self.cs.xh)


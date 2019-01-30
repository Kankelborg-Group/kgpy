
from unittest import TestCase
from copy import deepcopy
import quaternion

from kgpy.math import Vector, CoordinateSystem

__all__ = ['TestCoordinateSystem']


class TestCoordinateSystem(TestCase):

    def setUp(self):

        self.X = Vector([0, 0, 0])
        self.Q = quaternion.from_euler_angles(0, 0, 0)
        self.cs = CoordinateSystem(self.X, self.Q)

    def test_xh(self):

        # Check that the returned
        self.assertTrue(self.cs.xh == Vector([1, 0, 0]))

    def test__str__(self):

        # Check if returned value is of type string
        self.assertTrue(isinstance(self.cs.__str__(), str))

    def test__eq__(self):

        # Define first coordinate system
        v0 = Vector([0, 0, 0])
        q0 = quaternion.from_euler_angles(0, 0, 0)
        c0 = CoordinateSystem(v0, q0)

        # Define a second coordinate system the same as the first
        v1 = deepcopy(v0)
        q1 = deepcopy(q0)
        c1 = CoordinateSystem(v1, q1)

        # Check that the two coordinate systems are equal
        self.assertEqual(c0, c1)

        # Define a third coordinate system that has a different translation from the first
        v2 = Vector([1, 0, 0])
        q2 = quaternion.from_euler_angles(0, 0, 0)
        c2 = CoordinateSystem(v2, q2)

        # Check that these two coordinate systems are indeed not equal




    def test__add__(self):

        cs0 = self.cs + Vector([1, 0, 0])

        new_cs.Q.x = 1

        print(self.cs)
        print(new_cs)

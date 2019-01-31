
import pytest
from copy import deepcopy
from numbers import Real
import numpy as np
import quaternion

from kgpy.math import Vector, CoordinateSystem

__all__ = ['TestCoordinateSystem']

x_0 = (0, 1)
a_0 = (0, np.pi / 4)


@pytest.mark.parametrize('x', x_0)
@pytest.mark.parametrize('y', x_0)
@pytest.mark.parametrize('z', x_0)
@pytest.mark.parametrize('a', a_0)
@pytest.mark.parametrize('b', a_0)
@pytest.mark.parametrize('c', a_0)
class TestCoordinateSystem:

    def test_xh(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        X = Vector([x, y, z])
        Q = quaternion.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)



        # Check that the returned
        self.assertTrue(cs.xh == Vector([1, 0, 0]))

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

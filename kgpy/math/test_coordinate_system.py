
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

        # Create a test coordinate system
        X = Vector([x, y, z])
        Q = quaternion.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)

        # Check that x-hat is orthogonal to the other unit vectors
        assert cs.xh.dot(cs.yh) == pytest.approx(0.0)
        assert cs.xh.dot(cs.zh) == pytest.approx(0.0)


    def test__str__(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X = Vector([x, y, z])
        Q = quaternion.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)

        # Check if returned value is of type string
        assert isinstance(cs.__str__(), str)

    def test__eq__(self,  x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X0 = Vector([x, y, z])
        Q0 = quaternion.from_euler_angles(a, b, c)
        cs0 = CoordinateSystem(X0, Q0)

        # Define a second coordinate system the same as the first
        X1 = Vector([x, y, z])
        Q1 = quaternion.from_euler_angles(a, b, c)
        cs1 = CoordinateSystem(X1, Q1)

        # Check that the two coordinate systems are equal
        assert cs0 == cs1

        # Define a third coordinate system that is different from the first
        X2 = Vector([x + 1, y - 1, z])
        Q2 = quaternion.from_euler_angles(a + np.pi / 2, b - np.pi / 2, c)
        cs2 = CoordinateSystem(X2, Q2)

        # Check that these two coordinate systems are indeed not equal
        assert cs0 != cs2


    def test__add__(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X = Vector([x, y, z])
        Q = quaternion.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)

        # Create a test translation vector
        a = Vector([x + 1, y - 1, -z])

        # Execute the test addition between coordinate system and vector
        cs0 = cs + a

        # Check that the translation is what was expected
        assert cs0.X.x == X.x + a.x
        assert cs0.X.y == X.y + a.y
        assert cs0.X.z == X.z + a.z


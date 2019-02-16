
import pytest
from copy import deepcopy
from numbers import Real
import numpy as np
import quaternion as q
import astropy.units as u

from kgpy.math import Vector, CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs

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

    def test__init__(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X = Vector([x, y, z])
        Q = q.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)

    def test_xh(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X = Vector([x, y, z])
        Q = q.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)

        # Check that x-hat is orthogonal to the other unit vectors
        assert abs(cs.xh.dot(cs.yh)) < 1e-10
        assert abs(cs.xh.dot(cs.zh)) < 1e-10

    def test__str__(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X = Vector([x, y, z])
        Q = q.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)

        # Check if returned value is of type string
        assert isinstance(cs.__str__(), str)

    def test__eq__(self,  x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X0 = Vector([x, y, z])
        Q0 = q.from_euler_angles(a, b, c)
        cs0 = CoordinateSystem(X0, Q0)

        # Define a second coordinate system the same as the first
        X1 = Vector([x, y, z])
        Q1 = q.from_euler_angles(a, b, c)
        cs1 = CoordinateSystem(X1, Q1)

        # Check that the two coordinate systems are equal
        assert cs0 == cs1

        # Define a third coordinate system that is different from the first
        X2 = Vector([x + 1, y - 1, z])
        Q2 = q.from_euler_angles(a + np.pi / 2, b - np.pi / 2, c)
        cs2 = CoordinateSystem(X2, Q2)

        # Check that these two coordinate systems are indeed not equal
        assert cs0 != cs2

    def test__add__(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X = Vector([x, y, z])
        Q = q.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)

        # Create a test translation vector
        a = Vector([x + 1, y - 1, -z])

        # Execute the test addition between coordinate system and vector
        cs0 = cs + a

        # Check that the translation is what was expected
        assert cs0.X.x == X.x + a.x
        assert cs0.X.y == X.y + a.y
        assert cs0.X.z == X.z + a.z

    def test__radd__(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X = Vector([x, y, z])
        Q = q.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)

        # Create a test translation vector
        a = Vector([x + 1, y - 1, -z])

        # Execute the test addition between coordinate system and vector
        cs0 = a + cs

        # Check that the translation is what was expected
        assert cs0.X.x == X.x + a.x
        assert cs0.X.y == X.y + a.y
        assert cs0.X.z == X.z + a.z

    def test__mul__(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create a test coordinate system
        X = Vector([x, y, z])
        Q = q.from_euler_angles(a, b, c)
        cs = CoordinateSystem(X, Q)

        # Create a quaternion that will rotate the coordinate system back to the global coordinate system.
        Q2 = q.from_euler_angles(-c, -b, -a)    # type: q.quaternion

        # Execute the test multiplication operation
        cs2 = cs * Q2   # type: CoordinateSystem

        # If multiplication operation is correct, the quaternion attribute of the new coordinate system should be nearly
        # equal to the zero rotation quaternion
        assert np.isclose(cs2.Q, gcs().Q)

    def test__matmul__(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create first test coordinate system
        X1 = Vector([x, y, z] * u.mm)
        Q1 = q.from_euler_angles(a, b, c)
        cs1 = CoordinateSystem(X1, Q1)

        # Create second test coordinate system
        X2 = Vector([-x, -y, -z] * u.mm)
        Q2 = q.from_euler_angles(-c, -b, -a)
        cs2 = CoordinateSystem(X2, Q2, translation_first=False)

        # Execute test matmul operation
        cs3 = cs2 @ cs1

        # If the composition operator is correct, the resulting coordinate system should be equal to the global
        # coordinate system
        assert cs3.isclose(gcs())

    @pytest.mark.parametrize('tf', (True, False))
    def test_inverse(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real, tf: bool):

        # Create test coordinate system
        X1 = Vector([x, y, z] * u.mm)
        Q1 = q.from_euler_angles(a, b, c)
        cs1 = CoordinateSystem(X1, Q1, translation_first=tf)

        # Compose a coordinate system and its inverse
        cs2 = cs1 @ cs1.inverse
        cs3 = cs1.inverse @ cs1

        # Assert that the result is the global coordinate system
        assert cs2.isclose(gcs())
        assert cs3.isclose(gcs())

    @pytest.mark.parametrize('tf', (True, False))
    def test_diff(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real, tf: bool):

        # Create test coordinate system
        X1 = Vector([x, y, z] * u.mm)
        Q1 = q.from_euler_angles(a, b, c)
        cs1 = CoordinateSystem(X1, Q1, translation_first=tf)

        # Create a second test coordinate system
        # During testing, I found that this coordinate system needed to be independent of the first, to make non-
        # communative transformations.
        X2 = Vector([x + y, x - y, z] * u.mm)
        Q2 = q.from_euler_angles(a, 2 * b, 3 * c)
        cs2 = CoordinateSystem(X2, Q2)

        # Create two new coordinate systems that are multiples of the original coordinate system
        cs3 = cs2 @ cs1
        cs4 = cs1 @ cs2

        # Check that we can compute the difference between two coordinate systems
        assert cs3.diff(cs2).isclose(cs1)
        assert cs4.diff(cs1).isclose(cs2)

    def test_isclose(self, x: Real, y: Real, z: Real, a: Real, b: Real, c: Real):

        # Create first test coordinate system
        X1 = Vector([x, y, z] * u.mm)
        Q1 = q.from_euler_angles(a, b, c)
        cs1 = CoordinateSystem(X1, Q1)

        # Create second test coordinate system
        err = 1e-17
        X2 = Vector([x + err, y - err, z] * u.mm)
        Q2 = q.from_euler_angles(a + err, b - err, c)
        cs2 = CoordinateSystem(X2, Q2)

        # Check that the two coordinate systems are nearly equal
        assert cs1.isclose(cs2)

        # Create a third test coordinate system to verify that isclose returns False correctly
        X3 = Vector([x + 1, y - 1, z] * u.mm)
        Q3 = q.from_euler_angles(a + np.pi/4, b + np.pi/4, c + np.pi/4)
        cs3 = CoordinateSystem(X3, Q3)

        # Check that the first and third coordinate systems are not nearly equal
        assert not cs1.isclose(cs3)


def test_xy_intercept():

    # Define vectors to help manipulate coordinate systems
    z = Vector([0, 0, 1] * u.mm)
    x = Vector([1, 0, 0] * u.mm)

    # Define three test coordinate systems
    c1 = gcs() + z
    c2 = c1 + z
    c3 = c2 + z + x

    # Test that the intercepts for each coordinate system are computed correctly
    assert c1.xy_intercept(c2.X, c3.X) is None
    assert c2.xy_intercept(c1.X, c3.X) == Vector([0.5, 0, 2] * u.mm)
    assert c3.xy_intercept(c1.X, c2.X) is None
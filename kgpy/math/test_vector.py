
import pytest
from typing import List, Union
from numbers import Real
import numpy as np
import astropy.units as u

from kgpy.math import Vector

__all__ = ['TestVector']

t = (0, 1, 0.0, 1.0)


@pytest.mark.parametrize('x', t)
@pytest.mark.parametrize('y', t)
@pytest.mark.parametrize('z', t)
@pytest.mark.parametrize('unit', [
    1,
    u.mm
])
class TestVector:

    def test__init__(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):
        """
        Test the constructor by confirming that the Vector.X property is of the correct type, and that every component
        of the vector was set to the correct value.
        :param x: x-component of the Vector
        :param y: y-component of the Vector
        :param z: z-component of the Vector
        :param unit: units associated with the Vector
        :return: None
        """

        v = Vector([x, y, z] * unit)
        assert isinstance(v.X, u.Quantity)

        assert v.x == x * unit
        assert v.y == y * unit
        assert v.z == z * unit

    def test__eq__(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):
        """
        Test the equality operator of the Vector object.
        :param x: x-component of the Vector
        :param y: y-component of the Vector
        :param z: z-component of the Vector
        :param unit: units associated with the Vector
        :return: None
        """

        # Create two vectors with identical components
        v0 = Vector([x, y, z] * unit)
        v1 = Vector([x, y, z] * unit)

        # Check that the two vectors are equal
        assert v0 == v1

        # Create another vector with different components than the first two
        v2 = Vector([x + 1, y - 1, z] * unit)

        # Check that this vector is not equal to the original vector
        assert v0 != v2

        # If all three components of the Vector is zero check scalar equality
        if x == y == z:
            assert v0 == x * unit

    def test__add__(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):
        """
        Check the addition operator of the Vector object.
        :param x: x-component of the Vector
        :param y: y-component of the Vector
        :param z: z-component of the Vector
        :param unit: units associated with the Vector
        :return: None
        """

        # Declare two vectors to add together
        v0 = Vector([x, y, z] * unit)
        v1 = Vector([z, y, x] * unit)

        # Execute test addition operation
        v = v0 + v1

        # Check if the result of the addition operation is what we expected
        assert v.x == v0.x + v1.x
        assert v.y == v0.y + v1.y
        assert v.z == v0.z + v1.z

    @pytest.mark.parametrize('a', [
        1,
        0,
    ])
    def test__mul__(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity], a: Real):
        """
        Test the multiplication operator of the Vector object.
        :param x: x-component of the Vector
        :param y: y-component of the Vector
        :param z: z-component of the Vector
        :param unit: units associated with the Vector
        :param a: Factor to scale the Vector by.
        :return: None
        """

        # Append units onto the scalar factor.
        a = a * unit

        # Create vector
        v0 = Vector([x, y, z] * unit)

        # Execute test multiplication operation
        v = a * v0

        # Check that the scalar was multiplied by each component
        assert v.x == a * v0.x
        assert v.y == a * v0.y
        assert v.z == a * v0.z

        # Execute reverse multiplication test
        v = v0 * a

        # Check that the scalar was multiplied by each component
        assert v.x == a * v0.x
        assert v.y == a * v0.y
        assert v.z == a * v0.z

    def test_dot(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):

        # Create two test vectors
        a = Vector([x, y, z] * unit)
        b = Vector([x + z, y + x, z + y] * unit)

        # Test the dot product using the definition
        r = a.dot(b)
        assert r == a.x * b.x + a.y * b.y + a.z * b.z

    def test_cross(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):

        # Create two test vectors
        a = Vector([x, y, z] * unit)
        b = Vector([x + z, y + x, z + y] * unit)

        # Execute the test cross product
        v = a.cross(b)

        # Test the result using the definition of the cross product
        assert v.x == (a.y * b.z - a.z * b.y)
        assert v.y == -(a.x * b.z - a.z * b.x)
        assert v.z == (a.x * b.y - a.y * b.x)

    def test_mag(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):

        # Create test vector
        a = Vector([x, y, z] * unit)

        # Calculate the test magnitude
        m = a.mag

        # Compare to the definition of magnitude
        assert m == np.sqrt((x * x + y * y + z * z) * unit * unit)

    def test__array__(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):
        """
        Test the capability to cast a Vector to a numpy.ndarray. This operation is only allowed for dimensionless
        Vectors.
        :param x: x-component of the Vector
        :param y: y-component of the Vector
        :param z: z-component of the Vector
        :param unit: units associated with the Vector
        :return: None
        """

        # If the Vector is dimensionless, we expect to be able to cast it to an np.ndarray, otherwise we expect the
        # __array__ function to throw a type error.
        if isinstance(unit, Real):

            # Check that dimensionless vector casts to the same value as a numpy array
            v0 = Vector([x, y, z] * unit)
            a0 = v0.__array__()
            b0 = np.array([x, y, z])
            assert np.all(a0 == b0)

            # Check that we can make the above test fail by changing the value of the numpy array
            v0 = Vector([x, y, z] * unit)
            a0 = v0.__array__()
            b0 = np.array([x, y, z + 1])
            assert not np.all(a0 == b0)

        elif isinstance(unit, u.Quantity):

            # Check that converting a vector with dimensions throws a TypeError
            v0 = Vector([x, y, z] * u.m)
            with self.assertRaises(TypeError):
                v0.__array__()
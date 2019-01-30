
import pytest
from typing import List, Union
from numbers import Real
import numpy as np
import astropy.units as u

from kgpy.math import Vector

__all__ = ['TestVector']

t = (-1, 0, 10)


@pytest.mark.parametrize('x', t)
@pytest.mark.parametrize('y', t)
@pytest.mark.parametrize('z', t)
# @pytest.mark.parametrize('X0, X1', [
#     ([0, 0, 0], [0, 0, 1]),
#     ([1, 0, 0], [1, 1, 0]),
#     ([0, 1, 0], [2, 0, 1]),
#     ([1, 0, 1], [0, 3, 0]),
#     ([1, 2, 3], [3, 2, 1]),
# ])
@pytest.mark.parametrize('unit', [
    1,
    u.mm
])
class TestVector:

    def test__init__(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):

        v = Vector([x, y, z] * unit)
        assert isinstance(v.X, u.Quantity)

        assert v.x == x * unit
        assert v.y == y * unit
        assert v.z == z * unit

    def test__eq__(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):

        # Create two vectors with identical components
        v0 = Vector([x, y, z] * unit)
        v1 = Vector([x, y, z] * unit)

        # Check that the two vectors are equal
        assert v0 == v1

        # Create another vector with different components than the first two
        v2 = Vector([x + 1, y - 1, z] * unit)

        # Check that this vector is not equal to the original vector
        assert v0 != v2

    def test__add__(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity]):

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
        10,
        0,
        -1,
    ])
    def test__mul__(self, x: Real, y: Real, z: Real, unit: Union[Real, u.Quantity], a: Real):

        a = a * unit

        # Create vector
        v0 = Vector([x, y, z] * unit)

        # Execute test multiplication operation
        v = a * v0

        # Check that the scalar was multiplied by each component
        assert v.x == a * v0.x
        assert v.y == a * v0.y
        assert v.z == a * v0.z


    # def test__array__(self):
    #
    #     # Check that dimensionless vector casts to the same value as a numpy array
    #     v0 = Vector([1, 1, 1])
    #     a0 = v0.__array__()
    #     b0 = np.array([1, 1, 1])
    #     self.assertTrue(np.all(a0 == b0))
    #
    #     # Check that we can make the above test fail by changing the value of the numpy array
    #     v0 = Vector([1, 1, 1])
    #     a0 = v0.__array__()
    #     b0 = np.array([1, 1, 0])
    #     self.assertFalse(np.all(a0 == b0))
    #
    #     # Check that converting a vector with dimensions throws a TypeError
    #     v0 = Vector([1, 1, 1] * u.m)
    #     with self.assertRaises(TypeError):
    #         v0.__array__()

from unittest import TestCase
import numpy as np
import astropy.units as u

from kgpy.math import Vector

__all__ = ['TestVector']


class TestVector(TestCase):

    def test__init__(self):
        """
        Test constructor with list, array and quantity input parameters
        :return: None
        """

        # Test list input type
        v1 = Vector([1, 2, 3])
        self.assertIsInstance(v1.X, u.Quantity)

        # Test array input type
        v2 = Vector(np.array([1, 2, 3]))
        self.assertIsInstance(v2.X, u.Quantity)

        # Test Quantity input type
        v3 = Vector([1, 2, 3] * u.m)
        self.assertIsInstance(v3.X, u.Quantity)

    def test__eq__(self):

        # Declare vectors for equality testing
        v0 = Vector([1, 1, 1])
        v1 = Vector([1, 1, 1])
        v2 = Vector([1, 1, 0])

        # Assert vector is equal to itself
        self.assertEqual(v0, v1)

        # Assert two different vectors are not equal
        self.assertNotEqual(v0, v2)

    def test__add__(self):

        # Declare two vectors to add together
        v0 = Vector([1, 0, 0])
        v1 = Vector([0, 1, 0])

        # Declare vector that is the expected result of adding v0 and v1 together
        v2 = Vector([1, 1, 0])

        # Execute test addition operation
        v = v0 + v1

        # Check if the result of the addition operation is what we expected
        self.assertEqual(v, v2)

    def test__mul__(self):

        # Declare scalar and vector to multiply together
        a = 2 * u.m
        v0 = Vector([1, 1, 1])

        # Declare vector that is the expected result of multiplying a and v0 together
        v1 = Vector([2, 2, 2] * u.m)

        # Execute test multiplication operation
        v = a * v0

        # Check if the result is what we expected
        self.assertEqual(v, v1)

    def test__array__(self):

        # Check that dimensionless vector casts to the same value as a numpy array
        v0 = Vector([1, 1, 1])
        a0 = v0.__array__()
        b0 = np.array([1, 1, 1])
        self.assertTrue(np.all(a0 == b0))

        # Check that we can make the above test fail by changing the value of the numpy array
        v0 = Vector([1, 1, 1])
        a0 = v0.__array__()
        b0 = np.array([1, 1, 0])
        self.assertFalse(np.all(a0 == b0))

        # Check that converting a vector with dimensions throws a TypeError
        v0 = Vector([1, 1, 1] * u.m)
        with self.assertRaises(TypeError):
            v0.__array__()
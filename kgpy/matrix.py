"""
Complement to the :code:`kgpy.vector` package for matrices.
"""

import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['mul']


def transpose(a: u.Quantity) -> u.Quantity:
    return np.swapaxes(a, ~0, ~1)


def mul(a: u.Quantity, b: u.Quantity) -> u.Quantity:
    # a = transpose(a)
    a = np.expand_dims(a, ~0)
    # b = transpose(b.copy())
    b = np.expand_dims(b, ~2)
    return np.sum(a * b, axis=~1)

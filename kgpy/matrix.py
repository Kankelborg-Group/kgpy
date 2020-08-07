"""
Complement to the :code:`kgpy.vector` package for matrices.
"""

import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['mul']


@typ.overload
def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...


@typ.overload
def mul(a: u.Quantity, b: u.Quantity) -> u.Quantity:
    ...


def mul(a, b):
    a = np.expand_dims(a, ~0)
    b = np.swapaxes(b, ~0, ~1)
    b = np.expand_dims(b, ~2)
    return np.sum(a * b, axis=~1)

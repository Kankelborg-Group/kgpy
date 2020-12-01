"""
Complement to the :mod:`kgpy.vector` package for matrices.
"""

import typing as typ
import numpy as np
import astropy.units as u

__all__ = [
    'xx', 'xy', 'xz',
    'yx', 'yy', 'yz',
    'zx', 'zy', 'zz',
    'mul'
]


def transpose(a: u.Quantity) -> u.Quantity:
    return np.swapaxes(a, ~0, ~1)


def mul(a: u.Quantity, b: u.Quantity) -> u.Quantity:
    # a = transpose(a)
    a = np.expand_dims(a, ~0)
    # b = transpose(b.copy())
    b = np.expand_dims(b, ~2)
    return np.sum(a * b, axis=~1)


xx = [[1, 0, 0],
      [0, 0, 0],
      [0, 0, 0]] * u.dimensionless_unscaled

xy = [[0, 1, 0],
      [0, 0, 0],
      [0, 0, 0]] * u.dimensionless_unscaled

xz = [[0, 0, 1],
      [0, 0, 0],
      [0, 0, 0]] * u.dimensionless_unscaled

yx = [[0, 0, 0],
      [1, 0, 0],
      [0, 0, 0]] * u.dimensionless_unscaled

yy = [[0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]] * u.dimensionless_unscaled

yz = [[0, 0, 0],
      [0, 0, 1],
      [0, 0, 0]] * u.dimensionless_unscaled

zx = [[0, 0, 0],
      [0, 0, 0],
      [1, 0, 0]] * u.dimensionless_unscaled

zy = [[0, 0, 0],
      [0, 0, 0],
      [0, 1, 0]] * u.dimensionless_unscaled

zz = [[0, 0, 0],
      [0, 0, 0],
      [0, 0, 1]] * u.dimensionless_unscaled

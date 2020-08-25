"""
Package for easier manipulation of vectors than the usual numpy functions.
"""

import typing as typ
import numpy as np
import astropy.units as u
from . import matrix

__all__ = [
    'x', 'y', 'z',
    'ix', 'iy', 'iz',
    'xy',
    'x_hat', 'y_hat', 'z_hat',
    'dot', 'outer', 'matmul', 'lefmatmul', 'length', 'normalize', 'from_components', 'from_components_cylindrical'
]

ix = 0
iy = 1
iz = 2

x = ..., ix
y = ..., iy
z = ..., iz

xy = ..., slice(None, iz)

x_hat = np.array([1, 0, 0])
y_hat = np.array([0, 1, 0])
z_hat = np.array([0, 0, 1])


def to_3d(a: u.Quantity) -> u.Quantity:
    return from_components(a[x], a[y])


def dot(a: u.Quantity, b: u.Quantity, keepdims: bool = True) -> u.Quantity:
    return np.sum(a * b, axis=~0, keepdims=keepdims)


def outer(a: u.Quantity, b: u.Quantity) -> u.Quantity:
    a = np.expand_dims(a, ~1)
    b = np.expand_dims(b, ~0)
    return a * b


def matmul(a: u.Quantity, b: u.Quantity) -> u.Quantity:
    b = np.expand_dims(b, ~0)
    return matrix.mul(a, b)[..., 0]


def lefmatmul(a: u.Quantity, b: u.Quantity) -> u.Quantity:
    a = np.expand_dims(a, ~0)
    return matrix.mul(a, b)[..., 0]


def length(a: u.Quantity, keepdims: bool = True) -> u.Quantity:
    return np.sqrt(np.sum(np.square(a), axis=~0, keepdims=keepdims))


def normalize(a: u.Quantity, keepdims: bool = True) -> u.Quantity:
    return a / length(a, keepdims=keepdims)


def from_components(x: u.Quantity = 0, y: u.Quantity = 0, z: u.Quantity = 0, use_z: bool = True) -> u.Quantity:
    x, y, z = np.broadcast_arrays(x, y, z, subok=True)
    if use_z:
        return np.stack([x, y, z], axis=~0)
    else:
        return np.stack([x, y], axis=~0)


def from_components_cylindrical(r: u.Quantity = 0, phi: u.Quantity = 0, z: u.Quantity = 0) -> u.Quantity:
    return from_components(r * np.cos(phi), r * np.sin(phi), z)


# def rotate_x(vector: np.ndarray, angle: u.Quantity, inverse: bool = False) -> np.ndarray:
#     if inverse:
#         angle = -angle
#     r = np.zeros(angle.shape + (3, 3))
#     r[..., 0, 0] = 1
#     r[..., 1, 1] = np.cos(angle)
#     r[..., 1, 2] = np.sin(angle)
#     r[..., 2, 1] = -np.sin(angle)
#     r[..., 2, 2] = np.cos(angle)
#     return matmul(r, vector)
#
#
# def rotate_y(vector: np.ndarray, angle: u.Quantity, inverse: bool = False) -> np.ndarray:
#     if inverse:
#         angle = -angle
#     r = np.zeros(angle.shape + (3, 3))
#     r[..., 0, 0] = np.cos(angle)
#     r[..., 0, 2] = -np.sin(angle)
#     r[..., 1, 1] = 1
#     r[..., 2, 0] = np.sin(angle)
#     r[..., 2, 2] = np.cos(angle)
#     return matmul(r, vector)
#
#
# def rotate_z(vector: np.ndarray, angle: u.Quantity, inverse: bool = False) -> np.ndarray:
#     if inverse:
#         angle = -angle
#     r = np.zeros(angle.shape + (3, 3))
#     r[..., 0, 0] = np.cos(angle)
#     r[..., 0, 1] = np.sin(angle)
#     r[..., 1, 0] = -np.sin(angle)
#     r[..., 1, 1] = np.cos(angle)
#     r[..., 2, 2] = 1
#     return matmul(r, vector)
#
#
# def rotate(vector: np.ndarray, angles: u.Quantity, inverse: bool = False) -> np.ndarray:
#     if not inverse:
#         vector = rotate_x(vector, angles[x], inverse=inverse)
#         vector = rotate_y(vector, angles[y], inverse=inverse)
#         vector = rotate_z(vector, angles[z], inverse=inverse)
#     else:
#         vector = rotate_z(vector, angles[z], inverse=inverse)
#         vector = rotate_y(vector, angles[y], inverse=inverse)
#         vector = rotate_x(vector, angles[x], inverse=inverse)
#     return vector

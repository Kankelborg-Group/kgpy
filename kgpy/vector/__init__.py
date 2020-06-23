import typing as typ
import numpy as np
import astropy.units as u
from .. import matrix

__all__ = ['x', 'y', 'z', 'ix', 'iy', 'iz', 'dot', 'matmul', 'length', 'normalize', 'from_components']

ix = 0
iy = 1
iz = 2

x = ..., ix
y = ..., iy
z = ..., iz


def dot(a: np.ndarray, b: np.ndarray, keepdims: bool = True) -> np.ndarray:
    return np.sum(a * b, axis=~0, keepdims=keepdims)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.expand_dims(b, ~1)
    return matrix.mul(a, b)[..., 0]


def length(a: np.ndarray, keepdims: bool = True) -> np.ndarray:
    return np.sqrt(np.sum(np.square(a), axis=~0, keepdims=keepdims))


def normalize(a: np.ndarray, keepdims: bool = True) -> np.ndarray:
    return a / length(a, keepdims=keepdims)


def from_components(ax: np.ndarray = 0, ay: np.ndarray = 0, az: np.ndarray = 0) -> np.ndarray:
    ax, ay, az = np.broadcast_arrays(ax, ay, az, subok=True)
    return np.stack([ax, ay, az], axis=~0)


def rotate_x(vector: np.ndarray, angle: u.Quantity, inverse: bool = False) -> np.ndarray:
    if inverse:
        angle = -angle
    r = np.zeros(angle.shape + (3, 3))
    r[..., 0, 0] = 1
    r[..., 1, 1] = np.cos(angle)
    r[..., 1, 2] = np.sin(angle)
    r[..., 2, 1] = -np.sin(angle)
    r[..., 2, 2] = np.cos(angle)
    return matmul(r, vector)


def rotate_y(vector: np.ndarray, angle: u.Quantity, inverse: bool = False) -> np.ndarray:
    if inverse:
        angle = -angle
    r = np.zeros(angle.shape + (3, 3))
    r[..., 0, 0] = np.cos(angle)
    r[..., 0, 2] = -np.sin(angle)
    r[..., 1, 1] = 1
    r[..., 2, 0] = np.sin(angle)
    r[..., 2, 2] = np.cos(angle)
    return matmul(r, vector)


def rotate_z(vector: np.ndarray, angle: u.Quantity, inverse: bool = False) -> np.ndarray:
    if inverse:
        angle = -angle
    r = np.zeros(angle.shape + (3, 3))
    r[..., 0, 0] = np.cos(angle)
    r[..., 0, 1] = np.sin(angle)
    r[..., 1, 0] = -np.sin(angle)
    r[..., 1, 1] = np.cos(angle)
    r[..., 2, 2] = 1
    return matmul(r, vector)


def rotate(vector: np.ndarray, angles: u.Quantity, inverse: bool = False) -> np.ndarray:
    if not inverse:
        vector = rotate_x(vector, angles[x], inverse=inverse)
        vector = rotate_y(vector, angles[y], inverse=inverse)
        vector = rotate_z(vector, angles[z], inverse=inverse)
    else:
        vector = rotate_z(vector, angles[z], inverse=inverse)
        vector = rotate_y(vector, angles[y], inverse=inverse)
        vector = rotate_x(vector, angles[x], inverse=inverse)
    return vector

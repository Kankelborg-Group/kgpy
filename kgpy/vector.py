import typing as typ
import numpy as np
import astropy.units as u
from . import matrix

__all__ = ['x', 'y', 'z', 'ix', 'iy', 'iz','dot', 'matmul', 'length', 'normalize']

ix = 0
iy = 1
iz = 2

x = ..., ix
y = ..., iy
z = ..., iz


@typ.overload
def dot(a: np.ndarray, b: np.ndarray, keepdims: bool = True) -> np.ndarray:
    ...


@typ.overload
def dot(a: u.Quantity, b: u.Quantity, keepdims: bool = True) -> u.Quantity:
    ...


def dot(a, b, keepdims=True):
    return np.sum(a * b, axis=~0, keepdims=keepdims)


@typ.overload
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...


@typ.overload
def matmul(a: u.Quantity, b: u.Quantity) -> u.Quantity:
    ...


def matmul(a, b):
    b = np.expand_dims(b, ~1)
    return matrix.mul(a, b)[..., 0]


@typ.overload
def length(a: np.ndarray, keepdims: bool = True) -> np.ndarray:
    ...


@typ.overload
def length(a: u.Quantity, keepdims: bool = True) -> u.Quantity:
    ...


def length(a, keepdims=True):
    return np.sqrt(np.sum(np.square(a), axis=~0, keepdims=keepdims))


@typ.overload
def normalize(a: np.ndarray, keepdims: bool = True) -> np.ndarray:
    ...


@typ.overload
def normalize(a: u.Quantity, keepdims: bool = True) -> u.Quantity:
    ...


def normalize(a, keepdims=True):
    return a / length(a, keepdims=keepdims)


@typ.overload
def from_components(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    ...


@typ.overload
def from_components(ax: u.Quantity, ay: u.Quantity, az: u.Quantity) -> u.Quantity:
    ...


def from_components(ax, ay, az):
    ax, ay, az = np.broadcast_arrays(ax, ay, az)
    return np.stack([ax, ay, az], axis=~0)

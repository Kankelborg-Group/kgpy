import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['dot', 'length', 'normalize']


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
    return np.sum(a * b, axis=~0)


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
def from_components(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    ...


@typ.overload
def from_components(x: u.Quantity, y: u.Quantity, z: u.Quantity) -> u.Quantity:
    ...


def from_components(x, y, z):
    return np.stack([x, y, z], axis=~0)

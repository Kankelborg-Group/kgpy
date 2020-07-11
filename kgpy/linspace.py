import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['linspace', 'midspace']


@typ.overload
def linspace(start: u.Quantity, stop: u.Quantity, num: int, axis: int = 0) -> u.Quantity:
    ...


def linspace(start: np.ndarray, stop: np.ndarray, num: int, axis: int = 0) -> np.ndarray:
    if num == 1:
        return np.expand_dims((start + stop) / 2, axis=axis)
    else:
        return np.linspace(start=start, stop=stop, num=num, axis=axis)


def midspace(start: np.ndarray, stop: np.ndarray, num: int, axis: int = 0) -> np.ndarray:
    a = np.linspace(start=start, stop=stop, num=num + 1, axis=axis)
    i0 = [slice(None)] * a.ndim
    i1 = i0.copy()
    i0[axis] = slice(None, ~0)
    i1[axis] = slice(1, None)
    return (a[i0] + a[i1]) / 2

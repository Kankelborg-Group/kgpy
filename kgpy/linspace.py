import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['linspace', 'midspace']


@typ.overload
def linspace(start: u.Quantity, stop: u.Quantity, num: int, axis: int = 0) -> u.Quantity:
    ...


def linspace(start: np.ndarray, stop: np.ndarray, num: int, axis: int = 0) -> np.ndarray:
    """
    A modified version of :code:`numpy.linspace()` that returns a value in the center of the range between `start`
    and `stop` if `num == 1` unlike `numpy.linspace` which would just return `start`.
    This function is often helful when creating a grid.
    Sometimes you want to test with only a single element, but you want that element to be in the center of the range
    and not off to one side.

    :param start: The starting value of the sequence.
    :param stop: The end value of the sequence, must be broadcastable with `start`.
    :param num: Number of samples to generate for this sequence.
    :param axis: The axis in the result used to store the samples. The default is the first axis.
    :return: An array the size of the broadcasted shape of `start` and `stop` with an additional dimension of length
    `num`.
    """
    if num == 1:
        return np.expand_dims((start + stop) / 2, axis=axis)
    else:
        return np.linspace(start=start, stop=stop, num=num, axis=axis)


def midspace(start: np.ndarray, stop: np.ndarray, num: int, axis: int = 0) -> np.ndarray:
    """
    A modified version of `numpy.linspace` that selects cell centers instead of cell edges.

    :param start:
    :param stop:
    :param num:
    :param axis:
    :return:
    """
    a = np.linspace(start=start, stop=stop, num=num + 1, axis=axis)
    i0 = [slice(None)] * a.ndim
    i1 = i0.copy()
    i0[axis] = slice(None, ~0)
    i1[axis] = slice(1, None)
    return (a[i0] + a[i1]) / 2

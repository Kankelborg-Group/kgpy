import typing as typ
import numpy as np

__all__ = ['arg_percentile', 'intensity', 'shift', 'width', 'skew', 'first_four_moments']


def arg_percentile(cube: np.ndarray, percentile: float, axis: int = ~0) -> np.ndarray:
    mod_axis = axis % len(cube.shape)

    cs = np.nancumsum(cube, axis=axis)

    y = percentile * intensity(cube, axis=axis)
    x1 = np.argmax(cs > y, axis=axis)
    x0 = x1 - 1

    indices = np.indices(x1.shape)
    indices = np.expand_dims(indices, axis)

    x0 = np.expand_dims(x0, axis)
    x1 = np.expand_dims(x1, axis)

    indices_0 = list(indices)
    indices_1 = list(indices)

    if axis >= 0:
        indices_0.insert(axis, x0)
        indices_1.insert(axis, x1)
    else:
        indices_0.insert(mod_axis + 1, x0)
        indices_1.insert(mod_axis + 1, x1)

    y0 = cs[tuple(indices_0)]
    y1 = cs[tuple(indices_1)]

    x = (y - y0) / (y1 - y0) * (x1 - x0) + x0

    return x


def intensity(cube: np.ndarray, axis: int = ~0) -> np.ndarray:
    return np.nansum(cube, axis=axis, keepdims=True)


def shift(cube: np.ndarray, axis: int = ~0) -> np.ndarray:
    return arg_percentile(cube, 0.5, axis=axis)


def width(cube: np.ndarray, axis: int = ~0):
    p1 = arg_percentile(cube, 0.25, axis=axis)
    p3 = arg_percentile(cube, 0.75, axis=axis)
    return p3 - p1


def skew(cube: np.ndarray, axis: int = ~0) -> np.ndarray:
    p1 = arg_percentile(cube, 0.25, axis=axis)
    p2 = arg_percentile(cube, 0.50, axis=axis)
    p3 = arg_percentile(cube, 0.75, axis=axis)
    return p3 - 2 * p2 + p1


def first_four_moments(cube: np.ndarray, axis: int = ~0) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I0 = intensity(cube, axis=axis)
    I1 = shift(cube, axis=axis)
    I2 = width(cube, axis=axis)
    I3 = skew(cube, axis=axis)

    I0 = I0.flatten()
    I1 = I1.flatten()
    I2 = I2.flatten()
    I3 = I3.flatten()

    return I0, I1, I2, I3

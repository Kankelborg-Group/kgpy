import numpy as np

__all__ = ['linspace']


def linspace(start: np.ndarray, stop: np.ndarray, num: int, axis: int = 0) -> np.ndarray:
    if num == 1:
        return np.expand_dims((start + stop) / 2, axis=axis)
    else:
        return np.linspace(start=start, stop=stop, num=num, axis=axis)

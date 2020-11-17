import typing as typ
import numpy as np
import scipy.constants

__all__ = ['golden_section_search']


def golden_section_search(
        func: typ.Callable[[np.ndarray], np.ndarray],
        a: np.ndarray,
        b: np.ndarray,
        tolerance: float = 1e-6,
) -> np.ndarray:

    f_a, f_b = func(a), func(b)
    a = np.broadcast_to(a, f_a.shape, subok=True).copy()
    b = np.broadcast_to(b, f_b.shape, subok=True).copy()

    c, d = calc_update(a, b)

    while np.max(np.abs(c - d)) > tolerance:

        f_c, f_d = func(c), func(d)
        mask = f_c < f_d

        b[mask] = d[mask]
        a[~mask] = c[~mask]

        c, d = calc_update(a, b)

    return (a + b) / 2


def calc_update(a: np.ndarray, b: np.ndarray) -> typ.Tuple[np.ndarray, np.ndarray]:
    e = (b - a) / scipy.constants.golden_ratio
    c = b - e
    d = a + e
    return c, d

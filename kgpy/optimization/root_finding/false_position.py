import typing as typ
import numpy as np

__all__ = ['false_position']


def false_position(
        func: typ.Callable[[np.ndarray], np.ndarray],
        bracket_min: np.ndarray,
        bracket_max: np.ndarray,
        max_abs_error: float = 1e-9,
        max_iterations: int = 100,
):
    t0, t1 = bracket_min, bracket_max
    f0, f1 = func(t0), func(t1)

    t0 = np.broadcast_to(t0, f0.shape, subok=True).copy()
    t1 = np.broadcast_to(t1, f1.shape, subok=True).copy()

    for i in range(max_iterations):

        t2 = (t0 * f1 - t1 * f0) / (f1 - f0)
        f2 = func(t2)

        func_error = np.nanmax(np.abs(f2))
        if func_error < max_abs_error:
            return t2

        is_left = np.sign(f0) == np.sign(f2)
        is_right = ~is_left

        t0[is_left] = t2[is_left]
        t1[is_right] = t2[is_right]

        f0[is_left] = f2[is_left]
        f1[is_right] = f2[is_right]

    raise ValueError('Number of iterations exceeded')

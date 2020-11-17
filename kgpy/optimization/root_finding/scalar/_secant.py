import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['secant']


def secant(
        func: typ.Callable[[np.ndarray], np.ndarray],
        root_guess: np.ndarray = np.array(0),
        step_size: float = 1,
        max_abs_error: float = 1e-9,
        max_iterations: int = 100,
) -> np.ndarray:

    t0, t1 = root_guess - step_size, root_guess
    f0, f1 = func(t0), func(t1)

    t0 = np.broadcast_to(t0, f0.shape, subok=True).copy()
    t1 = np.broadcast_to(t1, f1.shape, subok=True).copy()

    for i in range(max_iterations):

        func_error = np.abs(f1)
        mask = func_error > max_abs_error
        if not mask.any():
            return t1

        df = f1 - f0

        t2 = (t0 * f1 - t1 * f0) / df
        f2 = func(t2)

        t0[mask] = t1[mask]
        t1[mask] = t2[mask]

        f0[mask] = f1[mask]
        f1[mask] = f2[mask]

    raise ValueError('Number of iterations exceeded')

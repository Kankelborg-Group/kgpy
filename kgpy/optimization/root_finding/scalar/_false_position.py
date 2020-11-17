import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['false_position']


@typ.overload
def false_position(
        func: typ.Callable[[u.Quantity], u.Quantity],
        bracket_min: u.Quantity,
        bracket_max: u.Quantity,
        max_abs_error: u.Quantity = 1e-9,
        max_iterations: int = 100,
) -> u.Quantity:
    ...


def false_position(
        func: typ.Callable[[np.ndarray], np.ndarray],
        bracket_min: np.ndarray,
        bracket_max: np.ndarray,
        max_abs_error: float = 1e-9,
        max_iterations: int = 100,
) -> np.ndarray:
    """
    The false position method (often known by its latin name, regula falsi) is a bracketed root-finding method that uses
    linear interpolation to iteratively approximate the root of a nonlinear function.
    This implementation is based on the `wikipedia <https://en.wikipedia.org/wiki/Regula_falsi>`_ page.


    Parameters
    ----------
    func
        Scalar function to be minimized.
    bracket_min
        Minimum value of the range to search for the root.
    bracket_max
        Maximum value of the range to search for the root.
    max_abs_error
        Maximum absolute error of the objective function at the root.
    max_iterations
        Maximum iterations to try before declaring non-convergence.

    Returns
    -------
    ~numpy.ndarray
        Root of the function

    Raises
    ------
    ValueError
        If the number of iterations is exceeded and ::`max_abs_error` has not been reached.
    """

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

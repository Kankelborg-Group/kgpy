import typing as typ
import numpy as np

__all__ = ['secant']


def secant(
        func: typ.Callable[[np.ndarray], np.ndarray],
        root_guess: np.ndarray = np.array(0),
        step_size: np.ndarray = np.array(1),
        max_abs_error: float = 1e-9,
        max_iterations: int = 100,
):

    x0 = root_guess - step_size
    x1 = root_guess + step_size

    i = 0
    while True:

        if i > max_iterations:
            raise ValueError('Max iterations exceeded')
        i += 1

        f0, f1 = func(x0), func(x1)
        # f0, f1 = np.expand_dims(f0, ~0), np.expand_dims(f1, ~0)

        df = f1 - f0

        current_error = np.max(np.abs(df))
        print(np.min(np.abs(df)), current_error)
        if current_error < max_abs_error:
            break

        mask = df == 0

        x2 = (x0 * f1 - x1 * f0) / df

        mask = np.broadcast_to(mask, x2.shape)
        x1 = np.broadcast_to(x1, x2.shape, subok=True)
        x2[mask] = x1[mask]

        x0 = x1
        x1 = x2

    return x1

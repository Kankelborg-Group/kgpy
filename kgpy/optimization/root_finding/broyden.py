import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import vector, matrix

__all__ = ['good_method']


def good_method(
        func: typ.Callable[[np.ndarray], np.ndarray],
        root_guess: np.ndarray = np.ndarray(0),
        step_size: np.ndarray = np.array(1),
        max_abs_error: float = 1e-9,
        max_iterations: int = 100,
):
    x0, x1 = root_guess - step_size, root_guess + step_size

    dx = x1 - x0

    f0 = func(x0)

    j0 = []
    for component_index in range(dx.shape[~0]):
        c = ..., slice(component_index, component_index + 1)
        x0_c, x1_c = np.zeros_like(x0), np.zeros_like(x1)
        x0_c[c], x1_c[c] = x0[c], x1[c]
        j0.append((func(x1_c) - func(x0_c)) / dx[c])
    j0 = np.stack(j0, axis=~0)

    inv_j0 = np.linalg.inv(j0)

    i = 0
    while True:

        if i > max_iterations:
            raise ValueError('Max iterations exceeded')
        i += 1

        f1 = func(x1)

        df = f1 - f0

        f1_mag = vector.length(f1, keepdims=False)
        converged = f1_mag < max_abs_error

        plt.imshow(converged.sum((~4, ~3, ~2)), vmin=0)
        plt.show()

        current_error = np.max(f1_mag)
        print(converged.sum())
        print('error = ', current_error.to(u.nm))
        if converged.all():
            break



        dx = x1 - x0

        # mask = df == 0

        jdf = vector.matmul(inv_j0, df)
        dxjdf = vector.dot(dx, jdf)
        mask = (dxjdf == 0)[..., 0]
        factor = (dx - jdf) / dxjdf
        # factor[mask, :] = 0
        inv_j1 = inv_j0 + vector.outer(factor, vector.lefmatmul(dx, inv_j0))
        # inv_j1 = inv_j0 + matrix.mul(vector.outer(factor, dx), inv_j0)

        x2 = x1 - vector.matmul(inv_j1, f1)
        x1 = np.broadcast_to(x1, x2.shape, subok=True)
        x2[converged, :] = x1[converged, :]

        x0 = x1
        x1 = x2
        f0 = f1
        inv_j0 = inv_j1

    return x1


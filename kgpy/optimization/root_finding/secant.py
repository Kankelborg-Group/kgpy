import typing as typ
import numpy as np
import astropy.units as u
import kgpy.vector

__all__ = ['secant']


def secant(
        func: typ.Callable[[np.ndarray], np.ndarray],
        root_guess: np.ndarray = np.array(0),
        step_size: np.ndarray = np.array(1),
        max_abs_error: float = 1e-9,
        max_iterations: int = 100,
):

    x0, x1 = root_guess - step_size, root_guess + step_size

    i = 0
    while True:

        if i > max_iterations:
            raise ValueError('Max iterations exceeded')
        i += 1

        f1 = func(x1)
        dx = x1 - x0
        mask = kgpy.vector.length(dx, keepdims=False) != 0

        f1_mag = kgpy.vector.length(f1, keepdims=False)
        converged = f1_mag < max_abs_error
        if converged.all():
            break

        j1 = []
        for component_index in range(dx.shape[~0]):
            c = ..., slice(component_index, component_index + 1)
            x0_c, x1_c = x1.copy(), x1.copy()
            x0_c[c], x1_c[c] = x1[c] - step_size[c], x1[c] + step_size[c]
            j1.append((func(x1_c) - func(x0_c)) / (2 * step_size[c]))

        jac = np.stack(j1, axis=~0)
        inv_jac = np.zeros_like(jac)
        det = np.linalg.det(jac)
        singular = det == 0
        inv_jac[~singular, :, :] = np.linalg.inv(jac[~singular, :, :])

        x2 = x1 - kgpy.vector.matmul(inv_jac, f1)

        x1 = np.broadcast_to(x1, x2.shape, subok=True)
        mask = np.broadcast_to(mask, x2[..., 0].shape)
        x2[~mask, :] = x1[~mask, :]

        x0 = x1
        x1 = x2

    return x1



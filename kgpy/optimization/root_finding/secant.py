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
    x0, x1 = root_guess, root_guess + step_size

    for i in range(max_iterations):

        f1 = func(x1)
        dx = x1 - x0
        mask = kgpy.vector.length(dx, keepdims=False) != 0

        f1_mag = kgpy.vector.length(f1, keepdims=False)
        converged = f1_mag < max_abs_error
        if converged.all():
            return x1

        j1 = []
        for component_index in range(dx.shape[~0]):
            c = ..., slice(component_index, component_index + 1)
            x1_c = x1.copy()
            x1_c[c] -= step_size[c]
            j1.append((f1 - func(x1_c)) / step_size[c])

        jac = np.stack(j1, axis=~0)
        inv_jac = np.zeros(jac.shape) << 1 / jac.unit
        det = np.linalg.det(jac)
        singular = det == 0
        inv_jac[~singular, :, :] = np.linalg.inv(jac[~singular, :, :])

        x2 = x1 - kgpy.vector.matmul(inv_jac, f1)

        x1 = np.broadcast_to(x1, x2.shape, subok=True)
        mask = np.broadcast_to(mask, x2[..., 0].shape)
        x2[~mask, :] = x1[~mask, :]

        x0 = x1
        x1 = x2

    raise ValueError('Max iterations exceeded')



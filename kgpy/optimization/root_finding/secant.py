import typing as typ
import numpy as np
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

    dx = x1 - x0

    f0 = []
    for component_index in range(dx.shape[~0]):
        c = ..., slice(component_index, component_index + 1)
        x0_c = np.zeros_like(x0)
        x0_c[c] = x0[c]
        f0.append(func(x0_c) / dx[c])
    f0 = np.stack(f0, axis=~0)

    i = 0
    while True:

        if i > max_iterations:
            raise ValueError('Max iterations exceeded')
        i += 1

        dx = x1 - x0

        mask = dx == 0

        current_error = np.max(kgpy.vector.length(dx))
        print('error', current_error)
        if current_error < max_abs_error:
            break

        f1 = []
        for component_index in range(dx.shape[~0]):
            c = ..., slice(component_index, component_index + 1)
            x1_c = np.zeros_like(x1)
            x1_c[c] = x1[c]
            f1.append(func(x1_c) / dx[c])
        f1 = np.stack(f1, axis=~0)
        jac = f1 - f0
        inv_jac = np.linalg.inv(jac)


        # print('x1', x1)
        # print('f1', f1)
        # print('dx', dx)
        # print('inv_jac', inv_jac)

        # x2 = (x0 * f1 - x1 * f0) / df
        x2 = x1 - kgpy.vector.matmul(inv_jac, func(x1))


        # fmask = np.broadcast_to(mask, f1.shape)
        # f0 = np.broadcast_to(f0, f1.shape, subok=True)
        # f0 = f0.copy()
        # f1[fmask] = f0[fmask]

        mask = np.broadcast_to(mask, x2.shape)
        x1 = np.broadcast_to(x1, x2.shape, subok=True)
        x1 = x1.copy()
        x2[mask] = x1[mask]

        # print('x2', x2)

        x0 = x1
        x1 = x2
        f0 = f1
        # x1[..., components] = x2[..., components]

        print()

    return x1



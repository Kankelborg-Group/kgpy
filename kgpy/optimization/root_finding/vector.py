import typing as typ
import numpy as np
import astropy.units as u
import kgpy.vector
import kgpy.matrix
import kgpy.units
from kgpy import vector, matrix

__all__ = ['secant_2d']


def secant_2d(
        func: typ.Callable[[vector.Vector2D], vector.Vector2D],
        root_guess: typ.Optional[vector.Vector2D] = None,
        step_size: u.Quantity = 1 * u.dimensionless_unscaled,
        max_abs_error: u.Quantity = 1e-9 * u.dimensionless_unscaled,
        max_iterations: int = 100,
        broydens_good_method: bool = False
) -> vector.Vector2D:

    if root_guess is None:
        root_guess = vector.Vector2D()

    x0, x1 = root_guess, root_guess + step_size

    for i in range(max_iterations):

        f1 = func(x1)
        dx = x1 - x0
        mask = dx.length != 0

        converged = f1.length < max_abs_error
        if converged.all():
            # print('num 2d secant iterations', i)
            return x1

        if (i == 0) or not broydens_good_method:

            x_step = x1 + step_size * vector.x_hat.xy
            y_step = x1 + step_size * vector.y_hat.xy
            jac_x = (func(x_step) - f1) / step_size
            jac_y = (func(y_step) - f1) / step_size
            jacobian = matrix.Matrix2D()
            jacobian.xx = jac_x.x
            jacobian.xy = jac_y.x
            jacobian.yx = jac_x.y
            jacobian.yy = jac_y.y


            # j1 = []
            # for component_index in range(dx.shape[~0]):
            #     c = ..., slice(component_index, component_index + 1)
            #     x1_c = x1.copy()
            #     x1_c[c] = x1_c[c] - step_size[c]
            #     j1_c = (f1 - func(x1_c)) / step_size[c]
            #     j1.append(j1_c)
            #
            # jac = np.stack(j1, axis=~0)
            # inv_jac = np.zeros_like(jac) * (1 / jac.unit**2)
            # det = np.linalg.det(jac)
            # singular = det == 0
            # inv = np.linalg.inv(jac[~singular, :, :])
            # inv_jac[~singular, :, :] = inv

        else:
            raise NotImplementedError
            # df = f1 - f0
            # jf = kgpy.vector.matmul(inv_jac, df)
            # cmat = kgpy.vector.outer((dx - jf) / kgpy.vector.dot(dx, jf), dx)
            # inv_jac = inv_jac + kgpy.matrix.mul(cmat, inv_jac)

        # x2 = x1 - kgpy.vector.matmul(inv_jac, f1)
        # correction = 0.9999 * kgpy.vector.matmul(inv_jac, f1)
        correction = 0.9999 * ~jacobian @ f1
        x2 = x1 - correction

        x1 = np.broadcast_to(x1, x2.shape, subok=True)
        mask = np.broadcast_to(mask, x2.shape, subok=True)
        x2[~mask] = x1[~mask]

        f0 = f1
        x0 = x1
        x1 = x2

    raise ValueError('Max iterations exceeded')



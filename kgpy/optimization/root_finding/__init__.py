import typing as typ
import numpy as np
import kgpy.labeled
import kgpy.vector

InputT = typ.TypeVar('InputT', bound=kgpy.vector.VectorLike)
OutputT = typ.TypeVar('OutputT', bound=kgpy.vector.VectorLike)


def secant(
        func: typ.Callable[[InputT], OutputT],
        root_guess: InputT,
        step_size: InputT,
        max_abs_error: OutputT,
        max_iterations: int = 100,
) -> InputT:

    x0 = root_guess - step_size
    x1 = root_guess

    for i in range(max_iterations):

        f1 = func(x1)

        if isinstance(f1, kgpy.vector.AbstractVector):
            root_error = f1.length
        else:
            root_error = np.abs(f1)

        if np.all(root_error < max_abs_error):
            print('max iteration', i)
            return x1

        dx = x1 - x0
        if isinstance(dx, kgpy.vector.AbstractVector):
            mask = dx.length != 0
        else:
            mask = dx != 0

        jacobian = 0 * f1 / step_size

        if isinstance(x1, kgpy.vector.AbstractVector):
            for component in x1.coordinates:
                x0_component = x1.copy_shallow()
                setattr(x0_component, component, x0_component.coordinates[component] - step_size.coordinates[component])
                df_component = f1 - func(x0_component)
                dfdx = type(x0)()
                setattr(dfdx, component, df_component / step_size.coordinates[component])
                jacobian = jacobian + dfdx

            if isinstance(f1, kgpy.vector.AbstractVector):
                jacobian = jacobian.to_matrix()

        else:
            df = f1 - func(x0)
            jacobian = jacobian + df / dx

        if isinstance(f1, kgpy.vector.AbstractVector) and isinstance(x1, kgpy.vector.AbstractVector):
            correction = 0.9999 * ~jacobian @ f1

        else:
            correction = 0.9999 * f1 / jacobian

        x2 = x1 - correction

        if not isinstance(x2, (int, float, complex,)) and x2.shape:
            if not isinstance(mask, kgpy.labeled.ArrayInterface):
                mask = kgpy.labeled.Array(mask)
            mask = np.broadcast_to(mask, x2.shape)
            x2 = np.broadcast_to(x2, x2.shape).copy()
            if i == 0:
                x1 = x1 + 0 * x2
            x2[~mask] = x1[~mask]

        x0 = x1
        x1 = x2

    raise ValueError('Max iterations exceeded')



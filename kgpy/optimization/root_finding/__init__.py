import typing as typ
import numpy as np
import kgpy.labeled
import kgpy.vectors

InputT = typ.TypeVar('InputT', bound=kgpy.vectors.VectorLike)
OutputT = typ.TypeVar('OutputT', bound=kgpy.vectors.VectorLike)


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

        if isinstance(f1, kgpy.vectors.AbstractVector):
            root_error = f1.length
        else:
            root_error = np.abs(f1)

        if np.all(root_error < max_abs_error):
            return x1

        dx = x1 - x0
        if isinstance(dx, kgpy.vectors.AbstractVector):
            dx = dx.length

        mask = dx != 0
        mask = mask & np.isfinite(dx)

        if np.all(x1 == x0):
            return x1

        if isinstance(f1, kgpy.vectors.AbstractVector) and isinstance(x1, kgpy.vectors.AbstractVector):

            jacobian = type(f1)()
            for component_f in jacobian.coordinates:
                jacobian.coordinates[component_f] = type(x1)()

            for component in x1.coordinates:
                x0_component = x1.copy_shallow()
                x0_component.coordinates[component] = x0_component.coordinates[component] - step_size.coordinates[component]
                f0_component = func(x0_component)
                df_component = f1 - f0_component
                for c in jacobian.coordinates:
                    jacobian.coordinates[c].coordinates[component] = df_component.coordinates[c] / step_size.coordinates[component]

            jacobian = jacobian.to_matrix()

            correction = 0.9999 * ~jacobian @ f1

            else:
                jacobian = type(x1)()

                for component in x1.coordinates:
                    x0_component = x1.copy_shallow()
                    x0_component.coordinates[component] = x0_component.coordinates[component] - step_size.coordinates[component]
                    f0_component = func(x0_component)
                    df_component = f1 - f0_component
                    jacobian.coordinates[component] = df_component / step_size.coordinates[component]

                correction = 0.9999 * f1 / jacobian

        else:
            df = f1 - func(x0)
            jacobian = df / dx
            correction = 0.9999 * f1 / jacobian

        # x2 = x1 - correction
        x2 = -correction + x1

        if not isinstance(x2, (int, float, complex,)) and x2.shape:
            if not isinstance(mask, kgpy.labeled.ArrayInterface):
                mask = kgpy.labeled.Array(mask)
            mask = np.broadcast_to(mask, x2.shape)
            x2 = np.broadcast_to(x2, x2.shape).copy()
            if i == 0:
                # x1 = x1 + 0 * x2
                x1 = 0 * x2 + x1
            x2[~mask] = x1[~mask]

        x0 = x1
        x1 = x2

    raise ValueError('Max iterations exceeded')



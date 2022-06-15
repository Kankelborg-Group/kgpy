import typing as typ
import numpy as np
import kgpy.labeled
import kgpy.vectors
import kgpy.matrix

InputT = typ.TypeVar('InputT', bound=kgpy.vectors.VectorLike)
OutputT = typ.TypeVar('OutputT', bound=kgpy.vectors.VectorLike)


def secant(
        func: typ.Callable[[InputT], OutputT],
        root_guess: InputT,
        step_size: InputT,
        max_abs_error: typ.Optional[OutputT] = None,
        max_iterations: int = 100,
) -> InputT:

    x0 = root_guess - step_size
    x1 = root_guess

    f0 = None

    for i in range(max_iterations):

        f1 = func(x1)

        if f0 is not None:
            if np.all(f1 == f0):
                return x1

        if max_abs_error is not None:
            root_error = np.abs(f1)
            if np.all(root_error < max_abs_error):
                return x1

        dx = x1 - x0
        if hasattr(dx, 'length'):
            mask = dx.length != 0
            mask = mask & np.isfinite(dx.length)
        else:
            mask = dx != 0
            mask = mask & np.isfinite(dx)

        if f0 is not None:
            mask = mask & (f1 != f0).array_labeled.any('component')

        if i > 0 and np.all(np.abs(dx) < step_size):
            return x1

        if np.all(x1 == x0):
            return x1

        if isinstance(x1, kgpy.vectors.AbstractVector):

            if isinstance(f1, kgpy.vectors.AbstractVector):

                jacobian = kgpy.matrix.CartesianND()
                for component_f in f1.coordinates:
                    jacobian.coordinates[component_f] = type(x1)()

                for component in x1.coordinates:
                    x0_component = x1.copy_shallow()
                    x0_component.coordinates[component] = x0_component.coordinates[component] - step_size.coordinates[component]
                    f0_component = func(x0_component)
                    df_component = f1 - f0_component
                    for c in jacobian.coordinates:
                        jacobian.coordinates[c].coordinates[component] = df_component.coordinates[c] / step_size.coordinates[component]

                correction = 0.9999 * ~jacobian @ f1
                print('correction',  correction)

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
            mask = mask & (df != 0)
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

        f0 = f1

    raise ValueError('Max iterations exceeded')



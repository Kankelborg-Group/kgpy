import typing as typ
import numpy as np
import matplotlib.pyplot as plt
from . import golden_section_search

__all__ = ['coordinate_descent']


def coordinate_descent(
        func: typ.Callable[[np.ndarray], np.ndarray],
        x_min: np.ndarray,
        x_max: np.ndarray,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        line_search_kwargs: typ.Dict[str, typ.Any] = None,
) -> np.ndarray:
    if line_search_kwargs is None:
        line_search_kwargs = {'tolerance': tolerance}

    x_min, x_max = np.broadcast_arrays(x_min.copy(), x_max.copy(), subok=True)

    x_current = x_min
    f_current = np.expand_dims(func(x_current), ~0)
    x_current, _ = np.broadcast_arrays(x_current, f_current, subok=True)
    x_current = x_current.copy()

    for i in range(max_iterations):

        for component_index in range(x_min.shape[~0]):

            component = ..., component_index

            def line_func(x_component: np.ndarray) -> np.ndarray:
                x_test = x_current.copy()
                x_test[component] = x_component
                return func(x_test)

            x_current[component] = golden_section_search(
                line_func, x_min[component], x_max[component], **line_search_kwargs)

        f_old = f_current
        f_current = func(x_current)
        current_error = np.max(np.abs(f_current - f_old))
        if current_error < tolerance:
            return x_current

    raise ValueError('Max iterations exceeded')
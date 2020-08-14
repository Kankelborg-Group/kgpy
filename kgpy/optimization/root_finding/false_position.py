import typing as typ
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['false_position']


def false_position(
        func: typ.Callable[[np.ndarray], np.ndarray],
        bracket_min: np.ndarray,
        bracket_max: np.ndarray,
        max_abs_error: float = 1e-9,
        max_iterations: int = 100,
):


    t0, t1 = bracket_min, bracket_max
    f0, f1 = func(t0), func(t1)

    t0 = np.broadcast_to(t0, f0.shape, subok=True).copy()
    t1 = np.broadcast_to(t1, f1.shape, subok=True).copy()

    for i in range(max_iterations):

        t2 = (t0 * f1 - t1 * f0) / (f1 - f0)
        f2 = func(t2)

        # print('t0', t0.squeeze())
        # print('t1', t1.squeeze())
        # print('t2', t2.squeeze())
        # print('f0', f0.squeeze())
        # print('f1', f1.squeeze())

        func_error = np.nanmax(np.abs(f2))
        # print('func_error', func_error)
        if func_error < max_abs_error:
            return t2

        is_left = np.sign(f0) == np.sign(f2)
        is_right = ~is_left

        t0[is_left] = t2[is_left]
        t1[is_right] = t2[is_right]

        f0[is_left] = f2[is_left]
        f1[is_right] = f2[is_right]

        # print()

    # t_test = np.linspace(bracket_min, bracket_max, 10000)
    # f_test = func(t_test)
    # t_test = np.broadcast_to(t_test, f_test.shape)
    # plt.plot(t_test.squeeze().T, f_test.squeeze().T)
    # plt.scatter(t0, f0)
    # plt.scatter(t1, f1)
    # plt.show()
    raise ValueError('Number of iterations exceeded')

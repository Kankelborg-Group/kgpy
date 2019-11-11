import typing as tp
import numpy as np
import numba

__all__ = ['percentile_stencil']


def percentile_stencil(percentile: float, kernel_size: tp.Tuple[int, ...]):

    sz_x = kernel_size[0]
    sz_y = kernel_size[1]
    sz_z = kernel_size[2]

    min_x = -(sz_x // 2)
    min_y = -(sz_y // 2)
    min_z = -(sz_z // 2)

    max_x = sz_x // 2 + 1
    max_y = sz_y // 2 + 1
    max_z = sz_z // 2 + 1

    kernel_limits = [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

    @numba.stencil(neighborhood=kernel_limits)
    def stencil(x: np.ndarray):

        y = np.empty(kernel_size)
        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                for k in range(min_z, max_z):
                    y[i, j, k] = x[i, j, k]

        return np.mean(y)

    return stencil


def percentile_filter(x: np.ndarray, percentile: float, kernel_size: tp.Tuple[int, ...]):
    sten = percentile_stencil(percentile, kernel_size)

    @numba.guvectorize([(numba.float32, numba.float32)], '() -> ()', target='cuda', nopython=True)
    def gufunc(a, out):
        out = 2 * a

    p = np.empty_like(x)
    gufunc(x, p)

    return p

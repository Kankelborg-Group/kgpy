import typing as tp
import numpy as np
import numba
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

__all__ = ['identify']


def identify(
        x: np.ndarray,
        axis: tp.Union[int, tp.Tuple[int, ...]] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, ...]] = 11,
        percentile_threshold: tp.Union[float, tp.Tuple[float, ...]] = 99,
        plot_histograms=False,
):
    pencil_size = 3
    nbins = 128

    if axis is None:
        axis = np.arange(x.ndim)
    else:
        axis = np.array(axis)

    kernel_size = np.array(kernel_size, dtype=np.int)
    percentile_threshold = np.array(percentile_threshold)

    if kernel_size.ndim == 0:
        kernel_size = np.array([kernel_size] * axis.size, dtype=np.int)

    if percentile_threshold.ndim == 0:
        percentile_threshold = np.array([percentile_threshold] * axis.size)

    for i, ax in enumerate(axis):

        k_sh = np.ones(x.ndim, dtype=np.int)
        k_sh[axis] = pencil_size
        k_sh[ax] = kernel_size[i]

        fx = scipy.ndimage.filters.percentile_filter(x, 50, size=k_sh)

        h_min = np.min(fx)
        # h_min = 0
        h_max = np.max(fx)
        hrange = [
            [h_min, h_max],
            [h_min, 2*h_max],
        ]
        hist, edges_x, edges_y = np.histogram2d(fx.reshape(-1), x.reshape(-1), bins=nbins, range=hrange)
        h_extent = [edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]]
        xgrid = edges_x[:-1] + ((edges_x[-1] - edges_x[0]) / len(edges_x)) / 2

        hist_sum = np.sum(hist, axis=-1, keepdims=True)
        hist /= hist_sum

        cs = np.cumsum(hist, axis=-1)

        x_thresh_upper = np.argmax(cs > (percentile_threshold[i] / 100), axis=-1)
        x_thresh_lower = np.argmax(cs > ((100 - percentile_threshold[i]) / 100), axis=-1)

        y_thresh_upper = edges_y[x_thresh_upper]
        y_thresh_lower = edges_y[x_thresh_lower]

        j = np.logical_and(hist_sum[..., 0] > 10, xgrid > 0)

        # a = np.array([xgrid[j] ** (1/2), xgrid[j] ** (2/2), xgrid[j] ** (3/2)]).T
        a = np.array([xgrid[j] ** 0, xgrid[j] ** 1, xgrid[j] ** 2, xgrid[j] ** 3, xgrid[j] ** 4]).T

        u = np.linalg.lstsq(a, y_thresh_upper[j])[0]
        v = np.linalg.lstsq(a, y_thresh_lower[j])[0]

        def cfunc(i_med: float, c: np.ndarray):
            # return c[0] * i_med ** (1 / 2) + c[1] * i_med ** (2 / 2) + c[2] * i_med ** (3 / 2)
            return c[0] * i_med ** 0 + c[1] * i_med ** 1 + c[2] * i_med ** 2 + c[3] * i_med ** 3 + c[4] * i_med ** 4

        def t_upper(i_med: float):
            return cfunc(i_med, u)

        def t_lower(i_med: float):
            return cfunc(i_med, v)

        plt.figure()
        plt.imshow(hist.T, norm=LogNorm(), origin='lower', extent=h_extent)
        plt.plot(xgrid, y_thresh_upper, scaley=False)
        plt.plot(xgrid, y_thresh_lower, scaley=False)
        plt.plot(xgrid, t_upper(xgrid), scaley=False)
        plt.plot(xgrid, t_lower(xgrid), scaley=False)
    





        



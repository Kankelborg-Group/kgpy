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

        # h_min = np.min(fx)
        h_min = 0
        h_max = np.max(fx)
        hrange = [
            [h_min, h_max],
            [h_min, h_max],
        ]
        hist, edges_x, edges_y = np.histogram2d(fx.flatten(), x.flatten(), bins=nbins, range=hrange)
        xgrid = edges_x[:-1]
        igrid = np.arange(len(xgrid))

        hist_sum = np.sum(hist, axis=-1, keepdims=True)
        hist /= hist_sum

        cs = np.cumsum(hist, axis=-1)

        x_thresh_upper = np.argmax(cs > (percentile_threshold[i] / 100), axis=-1)
        x_thresh_lower = np.argmax(cs > ((100 - percentile_threshold[i]) / 100), axis=-1)

        # ind = x_thresh_upper == x_thresh_lower
        # x_thresh_upper[ind] = igrid[ind]
        # x_thresh_lower[ind] = igrid[ind]

        y_thresh_upper = edges_y[x_thresh_upper]
        y_thresh_lower = edges_y[x_thresh_lower]

        j = hist_sum[..., 0] > 100

        a = np.array([xgrid[j] ** (1/2), xgrid[j] ** (2/2), xgrid[j] ** (3/2),]).T

        u = np.linalg.lstsq(a, y_thresh_upper[j])[0]
        v = np.linalg.lstsq(a, y_thresh_lower[j])[0]

        print(u.shape)
        print(v.shape)

        def t_upper(i_med: float):
            return u[0] * i_med ** (1/2) + u[1] * i_med ** (2/2) + u[2] * i_med ** (3/2)

        def t_lower(i_med: float):
            return v[0] * i_med ** (1/2) + v[1] * i_med ** (2/2) + v[2] * i_med ** (3/2)

        plt.figure()
        plt.imshow(hist.T, norm=LogNorm(), origin='lower', extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]])
        plt.plot(xgrid, y_thresh_upper)
        plt.plot(xgrid, y_thresh_lower)
        plt.plot(xgrid, t_upper(xgrid))
        plt.plot(xgrid, t_lower(xgrid))
    





        



import typing as tp
import numpy as np
import scipy.ndimage.filters
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

__all__ = ['identify', 'fix', 'identify_and_fix']


def identify_and_fix(
        data: np.ndarray,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, ...]] = 11,
        percentile_threshold: tp.Union[float, tp.Tuple[float, ...]] = 99,
        poly_deg: int = 1,
        num_hist_bins: int = 128,
        plot_histograms: bool = False,
) -> tp.Tuple[np.ndarray, np.ndarray]:

    mask = identify(data, axis, kernel_size, percentile_threshold, poly_deg, num_hist_bins, plot_histograms)

    fixed_data = fix(data, mask, axis)
    
    return fixed_data, mask


def identify(
        data: np.ndarray,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, ...]] = 11,
        percentile_threshold: tp.Union[float, tp.Tuple[float, ...]] = 99,
        poly_deg: int = 1,
        num_hist_bins: int = 128,
        plot_histograms: bool = False,
) -> np.ndarray:
    """
    Identify spikes in an image using a local median-dependent threshold.
    To start, this function calculates a histogram of local median vs value for every specified axis.
    From this histogram we can compute a cumulative distribution along the vertical (constant local median) axis.
    From the cumulative distribution, we can compute the location of the upper/lower threshold for a particular local
    median.
    The local median-dependent upper and lower thresholds are fit to a polynomial for interpolation/extrapolation.
    :param data: The input array
    :param axis: The axis or axes which are considered for computing the local median. If None, the median is computed
    along all the axes in `data`
    :param kernel_size: Int or list of ints the same length as `axis` specifying the size of the local median kernel.
    :param percentile_threshold: Float or list of floats specifying the percentile of the threshold for a particular
    value of the local median
    :param poly_deg:
    :param num_hist_bins:
    :param plot_histograms:
    :return:
    """
    pencil_size = 3
    spike_mask = np.zeros_like(data)
    
    if axis is None:
        axis = np.arange(data.ndim)
    else:
        axis = np.array(axis)
        if axis.ndim == 0:
            axis = np.array([axis])

    kernel_size = np.array(kernel_size, dtype=np.int)
    if kernel_size.ndim == 0:
        kernel_size = np.array([kernel_size] * axis.size, dtype=np.int)

    percentile_threshold = np.array(percentile_threshold)
    if percentile_threshold.ndim == 0:
        percentile_threshold = np.array([percentile_threshold] * axis.size)

    for i, ax in enumerate(axis):

        k_sh = np.ones(data.ndim, dtype=np.int)
        k_sh[axis] = pencil_size
        k_sh[ax] = kernel_size[i]

        fdata = scipy.ndimage.filters.percentile_filter(data, 50, size=k_sh)

        h_min = np.min(fdata)
        h_max = np.max(fdata)
        hrange = [
            [h_min, h_max],
            [h_min, 2*h_max],
        ]
        hist, edges_x, edges_y = np.histogram2d(fdata.reshape(-1), data.reshape(-1), bins=num_hist_bins, range=hrange)
        h_extent = [edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]]
        dx = (edges_x[-1] - edges_x[0]) / len(edges_x)
        xgrid = edges_x[:-1] + dx / 2

        hist_sum = np.sum(hist, axis=-1, keepdims=True)
        hist /= hist_sum

        cs = np.cumsum(hist, axis=-1)

        y_thresh_upper_ind = np.argmax(cs > (percentile_threshold[i] / 100), axis=-1) + 1
        y_thresh_lower_ind = np.argmax(cs > ((100 - percentile_threshold[i]) / 100), axis=-1)

        y_thresh_upper = edges_y[y_thresh_upper_ind]
        y_thresh_lower = edges_y[y_thresh_lower_ind]

        fit_weights = np.sqrt(np.sqrt(hist_sum[..., 0]))

        poly_coeff_upper = np.polyfit(xgrid, y_thresh_upper, deg=poly_deg, w=fit_weights)
        poly_coeff_lower = np.polyfit(xgrid, y_thresh_lower, deg=poly_deg, w=fit_weights)

        def t_upper(i_med: np.ndarray):
            return np.polyval(poly_coeff_upper, i_med)

        def t_lower(i_med: np.ndarray):
            return np.polyval(poly_coeff_lower, i_med)

        if plot_histograms:
            plt.figure()
            plt.imshow(hist.T, norm=LogNorm(), origin='lower', extent=h_extent)
            plt.plot(xgrid, y_thresh_upper, scaley=False)
            plt.plot(xgrid, y_thresh_lower, scaley=False)
            plt.plot(xgrid, t_upper(xgrid), scaley=False)
            plt.plot(xgrid, t_lower(xgrid), scaley=False)
            
        data_thresh_upper = t_upper(fdata)
        data_thresh_lower = t_lower(fdata)
        
        ind = np.logical_or(data_thresh_lower > data, data > data_thresh_upper)
        spike_mask[ind] += 1

    spike_mask = spike_mask == len(axis)
        
    return spike_mask


def fix(
        data: np.ndarray,
        mask: np.ndarray,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]] = None,
        kernel_size: int = 7,
):
    if axis is None:
        axis = np.arange(data.ndim)
    else:
        axis = np.array(axis)
    if axis.ndim == 0:
        axis = np.array([axis])

    kernel_size = np.array(kernel_size, dtype=np.int)
    if kernel_size.ndim == 0:
        kernel_size = np.array([kernel_size] * axis.size, dtype=np.int)

    fixed_data = data.copy()
    fixed_data[mask] = 0

    norm = np.ones_like(mask)
    norm[mask] = 0
    
    for i, ax in enumerate(axis):

        ksz = kernel_size[i]

        a = 0.05
        x = np.arange(ksz) - ksz // 2
        kernel = np.exp(-np.abs(x) / a)

        k_sh = np.ones(len(data.shape), dtype=np.int)
        k_sh[ax] = ksz
        kernel = np.reshape(kernel, k_sh)

        fixed_data = scipy.signal.convolve(fixed_data, kernel, mode='same')
        norm = scipy.signal.convolve(norm, kernel, mode='same')

    fixed_data /= norm
    fixed_data = np.nan_to_num(fixed_data, copy=False)
    fixed_data[~mask] = data[~mask]

    return fixed_data







        



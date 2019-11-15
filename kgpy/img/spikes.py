import typing as tp
import dataclasses
import numpy as np
import scipy.ndimage.filters
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from kgpy.plot.slice.image_stepper import CubeSlicer

__all__ = ['identify', 'fix', 'identify_and_fix']


@dataclasses.dataclass
class Stats:
    """
    A class representing a statistical model of spikes in an image.
    The statistical model consists of:
     - A 2D his
    """

    hist: np.ndarray
    hist_extent: np.ndarray
    x: np.ndarray
    thresh_upper: np.ndarray
    thresh_lower:  np.ndarray
    fit_weights: np.ndarray
    poly_deg: int = 1
    
    @property
    def thresh_fit_upper(self) -> tp.Callable:
        return self._thresh_fit(self.thresh_upper)
    
    @property
    def thresh_fit_lower(self) -> tp.Callable:
        return self._thresh_fit(self.thresh_lower)
    
    def _thresh_fit(self, thresh: np.ndarray) -> tp.Callable:
        interp = scipy.interpolate.interp1d(self.x, thresh)

        return np.poly1d(np.polyfit(self.x, thresh, deg=self.poly_deg, w=self.fit_weights))

    def plot(self):

        fig, ax = plt.subplots()

        self.plot_on_axes(ax)

    def plot_on_axes(self, ax: plt.Axes):
        # norm = None
        norm = LogNorm()
        ax.imshow(self.hist.T, norm=norm, origin='lower', extent=self.hist_extent)
        ax.plot(self.x, self.thresh_upper, 'b', scaley=False)
        ax.plot(self.x, self.thresh_lower, 'b', scaley=False)
        ax.plot(self.x, self.thresh_fit_upper(self.x), 'r', scaley=False)
        ax.plot(self.x, self.thresh_fit_lower(self.x), 'r', scaley=False)


@dataclasses.dataclass
class StatsList:
    data: tp.List[Stats]

    def plot(self):
        fig, ax = plt.subplots(1, len(self.data))

        for i, ast in enumerate(self.data):

            ast.plot_on_axes(ax[i])


def identify_and_fix(
        data: np.ndarray,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, ...]] = 11,
        percentile_threshold: tp.Union[float, tp.Tuple[float, ...]] = 99,
        poly_deg: int = 1,
        num_hist_bins: int = 128,
) -> tp.Tuple[np.ndarray, np.ndarray, StatsList]:

    mask, stats = identify(data, axis, kernel_size, percentile_threshold, poly_deg, num_hist_bins)

    fixed_data = fix(data, mask, axis)
    
    return fixed_data, mask, stats


def identify(
        data: np.ndarray,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, ...]] = 11,
        percentile_threshold: tp.Union[float, tp.Tuple[float, ...]] = 99,
        poly_deg: int = 1,
        num_hist_bins: int = 128,
) -> tp.Tuple[np.ndarray, StatsList]:
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
        
    axis_stats_list = []

    for i, ax in enumerate(axis):

        k_sh = np.ones(data.ndim, dtype=np.int)
        k_sh[axis] = pencil_size
        k_sh[ax] = kernel_size[i]

        fdata = scipy.ndimage.filters.percentile_filter(data, 50, size=k_sh)

        hist, hist_extent, xgrid, ygrid = calc_hist(fdata, data, num_hist_bins)

        hist_sum = np.sum(hist, axis=~0, keepdims=True)
        hist /= hist_sum

        cs = np.cumsum(hist, axis=~0)

        y_thresh_upper_ind = np.argmax(cs > (percentile_threshold[i] / 100), axis=-1) + 1
        y_thresh_lower_ind = np.argmax(cs > ((100 - percentile_threshold[i]) / 100), axis=-1)

        y_thresh_upper = ygrid[y_thresh_upper_ind]
        y_thresh_lower = ygrid[y_thresh_lower_ind]

        fit_weights = np.sqrt(hist_sum[..., 0])

        axis_stats = Stats(hist, hist_extent, xgrid, y_thresh_upper, y_thresh_lower, fit_weights=fit_weights,
                           poly_deg=poly_deg)
        axis_stats_list.append(axis_stats)

        # axis_stats.plot()
        # c = CubeSlicer(data)
        # plt.show()

        data_thresh_upper = axis_stats.thresh_fit_upper(fdata)
        data_thresh_lower = axis_stats.thresh_fit_lower(fdata)
        
        ind = np.logical_or(data_thresh_lower > data, data > data_thresh_upper)
        spike_mask[ind] += 1

    spike_mask = spike_mask == len(axis)
    
    stats = StatsList(axis_stats_list)
        
    return spike_mask, stats


def calc_hist(xdata: np.ndarray, ydata: np.ndarray, num_hist_bins: int):
    h_min = np.min(xdata)
    h_max = np.max(xdata)
    hrange = [
        [h_min, h_max],
        [h_min, 2 * h_max],
    ]
    
    hist, edges_x, edges_y = np.histogram2d(xdata.ravel(), ydata.ravel(), bins=num_hist_bins, range=hrange)
    hist_extent = np.array([edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]])
    
    dx = (edges_x[-1] - edges_x[0]) / len(edges_x)
    dy = (edges_y[-1] - edges_y[0]) / len(edges_y)
    
    xgrid = edges_x[:-1] + dx / 2
    ygrid = edges_y

    return hist, hist_extent, xgrid, ygrid


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


        



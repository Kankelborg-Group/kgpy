import typing as tp
import dataclasses
import numpy as np
import scipy.ndimage.filters
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import kgpy.filters

__all__ = ['identify', 'fix', 'identify_and_fix']


@dataclasses.dataclass
class Stats:
    """
    A class representing a statistical model of spikes in an image.
    """

    hist: np.ndarray
    hist_extent: np.ndarray
    x: np.ndarray
    thresh_pts_upper: np.ndarray
    thresh_pts_lower:  np.ndarray
    fit_weights: np.ndarray
    poly_deg: int = 1

    def thresh_upper(self, x: np.ndarray):
        return self._thresh(self.thresh_pts_upper, x)

    def thresh_lower(self, x: np.ndarray):
        return self._thresh(self.thresh_pts_lower, x)

    def _thresh(self, thresh_pts: np.ndarray, x: np.ndarray):

        use_fit_thresh = self.x[np.argmax(self.fit_weights < 1e9)]

        new_thresh_pts = np.empty_like(x)

        i = x <= use_fit_thresh
        j = x > use_fit_thresh

        new_thresh_pts[i] = np.interp(x[i], self.x, thresh_pts)
        new_thresh_pts[j] = self._thresh_fit(thresh_pts)(x[j])

        return new_thresh_pts
    
    def _thresh_fit(self, thresh_pts: np.ndarray) -> np.poly1d:

        return np.poly1d(np.polyfit(self.x, thresh_pts, deg=self.poly_deg, w=self.fit_weights))


def stats_list_plot(stats: tp.List[Stats], labels: tp.Optional[tp.List[str]] = None):

    fig, ax = plt.subplots(1, len(stats), figsize=(10, 4))   # type: plt.Figure, tp.List[plt.Axes]

    if isinstance(ax, plt.Axes):
        ax = [ax]

    for i, s in enumerate(stats):
        
        if i == len(stats) - 1:
            cbar = True
        else:
            cbar = False
            
        if labels is not None:
            label = labels[i]
        else:
            label = ''
            
        stats_plot(s, fig, ax[i], colorbar=cbar, label=label)


def stats_plot(stats: Stats, fig: plt.Figure, ax: plt.Axes, colorbar: bool = False, label=''):
    # norm = None
    norm = LogNorm()
    im = ax.imshow(stats.hist.T, norm=norm, origin='lower', extent=stats.hist_extent)
    if colorbar:
        fig.colorbar(im, ax=ax)

    ax.plot(stats.x, stats.thresh_pts_upper, 'k', scaley=False)
    ax.plot(stats.x, stats.thresh_pts_lower, 'k', scaley=False)
    ax.plot(stats.x, stats.thresh_upper(stats.x), 'r', scaley=False)
    ax.plot(stats.x, stats.thresh_lower(stats.x), 'r', scaley=False)
    ax.set_title(label)


def identify_and_fix(
        data: np.ndarray,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, ...]] = 11,
        percentile_threshold: tp.Union[float, tp.Tuple[float, float]] = 99,
        poly_deg: int = 1,
        num_hist_bins: int = 128,
        filter_type: str = "median"
) -> tp.Tuple[np.ndarray, np.ndarray, tp.List[Stats]]:

    mask, stats = identify(
        data=data,
        axis=axis,
        kernel_size=kernel_size,
        percentile_threshold=percentile_threshold,
        poly_deg=poly_deg,
        num_hist_bins=num_hist_bins,
        filter_type=filter_type
    )

    fixed_data, mask = fix(data, mask, axis)
    
    return fixed_data, mask, stats


def identify(
        data: np.ndarray,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, ...]] = 11,
        percentile_threshold: tp.Union[float, tp.Tuple[float, float]] = 99,
        poly_deg: int = 1,
        num_hist_bins: int = 128,
        filter_type: str = "median"
) -> tp.Tuple[np.ndarray, tp.List[Stats]]:
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
    :param percentile_threshold: Threshold at which to declare a spike for a particular value of the local median.
    This function can detect both positive and negative spikes, so we specify both an upper and lower threshold.
    If specified as a scalar, `percentile_threshold` is the upper threshold and the lower threshold is `
    100 - percentile_threshold`.
    If specified as a 2-element tuple, the first element is the lower threshold and the second element is the upper
    threshold.
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

    if np.array(percentile_threshold).ndim == 0:
        percentile_threshold = (100 - percentile_threshold, percentile_threshold)
        
    stats = []

    for i, ax in enumerate(axis):

        k_sh = np.ones(data.ndim, dtype=np.int)
        k_sh[axis] = pencil_size
        k_sh[ax] = kernel_size[i]

        if filter_type == "median":
            fdata = scipy.ndimage.median_filter(data, size=k_sh)
        elif filter_type == "trimmed_mean":
            fdata = kgpy.filters.mean_trimmed_numba(data, kernel_shape=k_sh)
        else:
            raise ValueError(f"filter type {filter_type} not recognized")

        hist, hist_extent, xgrid, ygrid = calc_hist(fdata, data, num_hist_bins)

        hist_sum = np.sum(hist, axis=~0, keepdims=True)
        hist /= hist_sum

        cs = np.cumsum(hist, axis=~0)

        y_thresh_upper_ind = np.argmax(cs > (percentile_threshold[1] / 100), axis=~0) + 1
        y_thresh_lower_ind = np.argmax(cs > (percentile_threshold[0] / 100), axis=~0)

        y_thresh_upper = ygrid[y_thresh_upper_ind]
        y_thresh_lower = ygrid[y_thresh_lower_ind]

        fit_weights = np.sqrt(hist_sum[..., 0])

        axis_stats = Stats(hist, hist_extent, xgrid, y_thresh_upper, y_thresh_lower, fit_weights=fit_weights,
                           poly_deg=poly_deg)
        stats.append(axis_stats)

        data_thresh_upper = axis_stats.thresh_upper(fdata)
        data_thresh_lower = axis_stats.thresh_lower(fdata)
        
        ind = np.logical_or(data_thresh_lower > data, data > data_thresh_upper)
        spike_mask[ind] += 1

    spike_mask = spike_mask == len(axis)
        
    return spike_mask, stats


def calc_hist(xdata: np.ndarray, ydata: np.ndarray, num_hist_bins: int):
    h_min = np.min(xdata)
    h_max = np.max(xdata)
    hrange = [
        [h_min, h_max],
        [np.min(ydata), 2 * h_max],
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
) -> tp.Tuple[np.ndarray, np.ndarray]:
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

    norm = np.ones_like(mask, dtype=np.float)
    norm[mask] = 0
    
    for i, ax in enumerate(axis):

        ksz = kernel_size[i]

        a = 0.5
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
    spikes = np.zeros_like(mask, dtype=np.float)
    spikes[mask] = data[mask]

    return fixed_data, spikes


        



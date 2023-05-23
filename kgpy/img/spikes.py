import typing as tp
import functools
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

    histogram: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    percentile_lower: float
    percentile_upper: float

    @property
    def x(self):
        return (self.x_edges[1:] + self.x_edges[:~0]) / 2

    @property
    def y(self):
        return (self.y_edges[1:] + self.y_edges[:~0]) / 2

    @property
    def histogram_normalized(self) -> np.ndarray:
        hist = self.histogram
        hist_sum = np.sum(hist, axis=~0, keepdims=True)
        hist /= hist_sum
        hist = np.nan_to_num(hist)
        return hist

    @property
    def histogram_cumsum(self) -> np.ndarray:
        return np.cumsum(self.histogram_normalized, axis=~0)

    @functools.cached_property
    def _thresh(self):

        hist_cumsum = self.histogram_cumsum

        thresh = scipy.interpolate.LinearNDInterpolator(
            points=np.stack(
                arrays=[
                    np.broadcast_to(self.x[:, np.newaxis], hist_cumsum.shape).reshape(-1),
                    hist_cumsum.reshape(-1)
                ],
                axis=~0,
            ),
            values=np.broadcast_to(self.y[np.newaxis, :], hist_cumsum.shape).reshape(-1),
            fill_value=0,
        )

        return thresh

    def thresh_upper(self, x: np.ndarray):
        return self._thresh(x, self.percentile_upper / 100)

    def thresh_lower(self, x: np.ndarray):
        return self._thresh(x, self.percentile_lower / 100)

    def plot(
            self,
            ax: plt.Axes,
            label: str = ''
    ):
        norm = LogNorm()
        img = ax.pcolormesh(
            *np.broadcast_arrays(self.x_edges[:, np.newaxis], self.y_edges[np.newaxis, :]),
            self.histogram_normalized,
            norm=norm,
        )

        x = np.linspace(self.x_edges.min(), self.x_edges.max(), num=1001)

        ax.plot(x, self.thresh_upper(x), color="black")
        ax.plot(x, self.thresh_lower(x), color="black")

        ax.set_title(label)


def stats_list_plot(stats: tp.List[Stats], labels: tp.Optional[tp.List[str]] = None):

    fig, ax = plt.subplots(1, len(stats), figsize=(10, 4), constrained_layout=True, squeeze=False)
    if isinstance(ax, plt.Axes):
        ax = [ax]

    for i, s in enumerate(stats):
            
        if labels is not None:
            label = labels[i]
        else:
            label = ''
            
        s.plot(ax=ax[0, i], label=label)


def identify_and_fix(
        data: np.ndarray,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, ...]] = 11,
        percentile_threshold: tp.Union[float, tp.Tuple[float, float]] = 99,
        num_hist_bins: int = 128,
        filter_type: str = "median"
) -> tp.Tuple[np.ndarray, np.ndarray, tp.List[Stats]]:

    mask, stats = identify(
        data=data,
        axis=axis,
        kernel_size=kernel_size,
        percentile_threshold=percentile_threshold,
        num_hist_bins=num_hist_bins,
        filter_type=filter_type
    )

    fixed_data, mask = fix(data, mask, axis)
    
    return fixed_data, mask, stats


def identify(
        data: np.ndarray,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, ...]] = 11,
        pencil_size: int = 3,
        percentile_threshold: tp.Union[float, tp.Tuple[float, float]] = 99,
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
            fdata = kgpy.filters.mean_trimmed_numba(data, kernel_shape=tuple(k_sh))
        else:
            raise ValueError(f"filter type {filter_type} not recognized")

        hist, x_edges, y_edges = calc_hist(
            xdata=fdata,
            ydata=data,
            num_hist_bins=num_hist_bins,
        )

        stats_axis = Stats(
            histogram=hist,
            x_edges=x_edges,
            y_edges=y_edges,
            percentile_lower=percentile_threshold[0],
            percentile_upper=percentile_threshold[1],
        )
        stats.append(stats_axis)

        data_thresh_upper = stats_axis.thresh_upper(fdata)
        data_thresh_lower = stats_axis.thresh_lower(fdata)
        
        ind = np.logical_or(data_thresh_lower > data, data > data_thresh_upper)
        spike_mask[ind] += 1

    spike_mask = spike_mask == len(axis)
        
    return spike_mask, stats


def calc_hist(xdata: np.ndarray, ydata: np.ndarray, num_hist_bins: int):

    x_edges = np.geomspace(1, xdata.max(), num=num_hist_bins + 1)
    y_edges = np.geomspace(1, ydata.max(), num=num_hist_bins + 1)

    x_edges = np.concatenate([-x_edges[::-1], x_edges])
    y_edges = np.concatenate([-y_edges[::-1], y_edges])

    x_edges = x_edges[x_edges >= xdata.min()]
    y_edges = y_edges[y_edges >= ydata.min()]

    hist, x_edges, y_edges = np.histogram2d(
        x=xdata.reshape(-1),
        y=ydata.reshape(-1),
        bins=(x_edges, y_edges),
    )

    return hist, x_edges, y_edges


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


        



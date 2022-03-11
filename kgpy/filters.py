import typing as typ
import numpy as np
import numpy.typing
import scipy.stats
import matplotlib.pyplot as plt
import astropy.units as u

__all__ = [
    'gaussian_trimmed',
    'mean_trimmed',
]


def gaussian_trimmed(
        array: numpy.typing.ArrayLike,
        kernel_size: typ.Union[int, typ.Sequence[int]] = 11,
        kernel_width: numpy.typing.ArrayLike = 3,
        proportion: u.Quantity = 25 * u.percent,
):
    ndim = array.ndim
    if np.isscalar(kernel_size):
        kernel_size = (kernel_size, ) * ndim
    kernel_size = np.array(kernel_size)

    if np.isscalar(kernel_width):
        kernel_width = (kernel_width, ) * ndim
    kernel_width = np.array(kernel_width)

    kernel_left = np.ceil(kernel_size / 2 - 1).astype(int)
    kernel_right = np.floor(kernel_size / 2).astype(int)

    proportion = proportion.to(u.percent).value

    array_padded = np.pad(
        array=array,
        pad_width=np.stack([kernel_left, kernel_right], axis=~0),
        mode='constant',
        constant_values=np.nan,
    )

    array_windowed = np.lib.stride_tricks.sliding_window_view(
        x=array_padded,
        window_shape=kernel_size,
        subok=True,
    ).copy()

    axes_kernel = tuple(~np.arange(ndim))
    limit_lower = np.nanpercentile(
        a=array_windowed,
        q=proportion,
        axis=axes_kernel,
        keepdims=True
    )
    limit_upper = np.nanpercentile(
        a=array_windowed,
        q=100 - proportion,
        axis=axes_kernel,
        keepdims=True
    )

    mask_lower = limit_lower < array_windowed
    mask_upper = array_windowed < limit_upper
    mask = mask_lower & mask_upper

    array_windowed[~mask] = np.nan
    array_windowed[~mask] = np.nan

    kernel = 1
    coordinates = np.indices(kernel_size) - kernel_size[(Ellipsis, ) + ndim * (np.newaxis, )] / 2
    for d in range(ndim):
        kernel = kernel * np.exp(-np.square(coordinates[d] / kernel_width[d]) / 2)

    return np.sum(array_windowed * kernel, axis=axes_kernel, where=mask) / np.sum(mask * kernel, axis=axes_kernel, where=mask)


def mean_trimmed(
        array: numpy.typing.ArrayLike,
        kernel_size: typ.Union[int, typ.Sequence[int]],
        proportion: u.Quantity = 25 * u.percent,
):
    ndim = array.ndim
    if np.isscalar(kernel_size):
        kernel_size = (kernel_size, ) * ndim
    kernel_size = np.array(kernel_size)

    kernel_left = np.ceil(kernel_size / 2 - 1).astype(int)
    kernel_right = np.floor(kernel_size / 2).astype(int)

    proportion = float(proportion)

    array_padded = np.pad(
        array=array,
        pad_width=np.stack([kernel_left, kernel_right], axis=~0),
        mode='constant',
        constant_values=np.nan,
    )

    array_windowed = np.lib.stride_tricks.sliding_window_view(
        x=array_padded,
        window_shape=kernel_size,
        subok=True
    )

    array_windowed = array_windowed.reshape(array.shape + (-1,))

    result = scipy.stats.trim_mean(
        a=array_windowed,
        proportiontocut=proportion,
        axis=~0,
    )

    return result


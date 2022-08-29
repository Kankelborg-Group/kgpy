from __future__ import annotations
import typing as typ
import numpy as np
import numpy.typing
import scipy.stats
import matplotlib.pyplot as plt
import numba
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

    return np.nansum(array_windowed * kernel, axis=axes_kernel) / np.nansum(mask * kernel, axis=axes_kernel)


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
        mode='reflect',
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

def median_numba(
        array: np.ndarray,
        kernel_shape: int | tuple[int, ...],
        axis: None | int | tuple[int, ...] = None,
):
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape, )

    if axis is None:
        axis = tuple(range(array.ndim))

    if isinstance(axis, int):
        axis = (axis, )

    axis_orthogonal = tuple(ax for ax in range(array.ndim) if ax not in axis)

    shape_orthogonal = tuple(sz for sz in np.array(array.shape)[np.array(axis_orthogonal)])

    result = np.zeros_like(array)

    for index in np.ndindex(*shape_orthogonal):

        slices = tuple(index[axis_orthogonal.index(ax)] if ax in axis_orthogonal else slice(None) for ax in range(array.ndim))

        if len(axis) == 1:
            result[slices] = _median_numba_1d(array[slices], kernel_shape=kernel_shape)

        elif len(axis) == 2:
            result[slices] = _median_numba_2d(array[slices], kernel_shape=kernel_shape)

        elif len(axis) == 3:
            res = _median_numba_3d(array[slices], kernel_shape=kernel_shape)
            result[slices] = res

        else:
            raise ValueError('Too many axis parameters, only 1-3 reduction axes are supported.')

    return result


@numba.njit(parallel=True)
def _median_numba_1d(
        array: np.ndarray,
        kernel_shape: tuple[int] = (11,),
):
    result = np.empty_like(array)

    array_shape_x, = array.shape

    kernel_shape_x, = kernel_shape

    for index_array_x in numba.prange(array_shape_x):

        values = np.empty(kernel_shape)

        for index_kernel_x in numba.prange(kernel_shape_x):

            position_kernel_x = index_kernel_x - kernel_shape_x // 2

            index_final_x = index_array_x + position_kernel_x

            index_final_x = np.abs(index_final_x)

            if index_final_x >= array_shape_x:
                index_final_x = ~(index_final_x % array_shape_x)

            values[index_kernel_x,] = array[index_final_x,]

        result[index_array_x,] = np.median(values)

    return result


@numba.njit(parallel=True)
def _median_numba_2d(
        array: np.ndarray,
        kernel_shape: tuple[int, int] = (11, 11),
):
    result = np.empty_like(array)

    array_shape_x, array_shape_y = array.shape

    kernel_shape_x, kernel_shape_y = kernel_shape

    for index_array_x in numba.prange(array_shape_x):
        for index_array_y in numba.prange(array_shape_y):

            values = np.empty(kernel_shape)

            for index_kernel_x in numba.prange(kernel_shape_x):
                for index_kernel_y in numba.prange(kernel_shape_y):

                    position_kernel_x = index_kernel_x - kernel_shape_x // 2
                    position_kernel_y = index_kernel_y - kernel_shape_y // 2

                    index_final_x = index_array_x + position_kernel_x
                    index_final_y = index_array_y + position_kernel_y

                    index_final_x = np.abs(index_final_x)
                    index_array_y = np.abs(index_array_y)

                    if index_final_x >= array_shape_x:
                        index_final_x = ~(index_final_x % array_shape_x)

                    if index_final_y >= array_shape_y:
                        index_final_y = ~(index_final_y % array_shape_y)

                    values[index_kernel_x, index_kernel_y] = array[index_final_x, index_final_y]

            result[index_array_x, index_array_y] = np.median(values)

    return result


@numba.njit(parallel=True,)
def _median_numba_3d(
        array: np.ndarray,
        kernel_shape: tuple[int, int, int] = (11, 11, 11),
):
    result = np.empty_like(array)

    array_shape_x, array_shape_y, array_shape_z = array.shape

    kernel_shape_x, kernel_shape_y, kernel_shape_z = kernel_shape

    for index_array_x in numba.prange(array_shape_x):
        for index_array_y in numba.prange(array_shape_y):
            for index_array_z in numba.prange(array_shape_z):

                values = np.empty(kernel_shape)

                for index_kernel_x in numba.prange(kernel_shape_x):
                    for index_kernel_y in numba.prange(kernel_shape_y):
                        for index_kernel_z in numba.prange(kernel_shape_z):

                            position_kernel_x = index_kernel_x - kernel_shape_x // 2
                            position_kernel_y = index_kernel_y - kernel_shape_y // 2
                            position_kernel_z = index_kernel_z - kernel_shape_z // 2

                            index_final_x = index_array_x + position_kernel_x
                            index_final_y = index_array_y + position_kernel_y
                            index_final_z = index_array_z + position_kernel_z

                            index_final_x = np.abs(index_final_x)
                            index_array_y = np.abs(index_array_y)
                            index_array_z = np.abs(index_array_z)

                            if index_final_x >= array_shape_x:
                                index_final_x = ~(index_final_x % array_shape_x)

                            if index_final_y >= array_shape_y:
                                index_final_y = ~(index_final_y % array_shape_y)

                            if index_final_z >= array_shape_z:
                                index_final_z = ~(index_final_z % array_shape_z)

                            values[index_kernel_x, index_kernel_y, index_kernel_z] = array[index_final_x, index_final_y, index_final_z]

                result[index_array_x, index_array_y, index_array_z] = np.median(values)

    return result


def mean_trimmed_numba(
        array: np.ndarray,
        kernel_shape: int | tuple[int, ...],
        proportion: float = 0.25,
        axis: None | int | tuple[int, ...] = None,
):
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape,)

    if axis is None:
        axis = tuple(range(array.ndim))

    if isinstance(axis, int):
        axis = (axis,)

    axis_orthogonal = tuple(ax for ax in range(array.ndim) if ax not in axis)

    shape_orthogonal = tuple(sz for sz in np.array(array.shape)[np.array(axis_orthogonal)])

    result = np.zeros_like(array)

    for index in np.ndindex(*shape_orthogonal):

        slices = tuple(
            index[axis_orthogonal.index(ax)] if ax in axis_orthogonal else slice(None) for ax in range(array.ndim))

        if len(axis) == 1:
            result[slices] = _mean_trimmed_numba_1d(array[slices], kernel_shape=kernel_shape, proportion=proportion)

        elif len(axis) == 2:
            result[slices] = _mean_trimmed_numba_2d(array[slices], kernel_shape=kernel_shape, proportion=proportion)

        elif len(axis) == 3:
            res = _mean_trimmed_numba_3d(array[slices], kernel_shape=kernel_shape, proportion=proportion)
            result[slices] = res

        else:
            raise ValueError('Too many axis parameters, only 1-3 reduction axes are supported.')

    return result


@numba.njit(parallel=True)
def _mean_trimmed_numba_1d(
        array: np.ndarray,
        kernel_shape: tuple[int] = (11,),
        proportion: float = 0.25,
):
    result = np.empty_like(array)

    array_shape_x, = array.shape

    kernel_shape_x, = kernel_shape

    for index_array_x in numba.prange(array_shape_x):

        values = np.empty(kernel_shape)

        for index_kernel_x in numba.prange(kernel_shape_x):

            position_kernel_x = index_kernel_x - kernel_shape_x // 2

            index_final_x = index_array_x + position_kernel_x

            index_final_x = np.abs(index_final_x)

            if index_final_x >= array_shape_x:
                index_final_x = ~(index_final_x % array_shape_x)

            values[index_kernel_x,] = array[index_final_x,]

        thresh_min = np.percentile(values, proportion)
        thresh_max = np.percentile(values, 1 - proportion)

        sum_convol = 0
        num_convol = 0
        for index_kernel_x in numba.prange(kernel_shape_x):
            value = values[index_kernel_x,]
            if thresh_min < value < thresh_max:
                sum_convol += value
                num_convol += 1

        result[index_array_x,] = sum_convol / num_convol

    return result


@numba.njit(parallel=True)
def _mean_trimmed_numba_2d(
        array: np.ndarray,
        kernel_shape: tuple[int, int] = (11, 11),
        proportion: float = 0.25,
):
    result = np.empty_like(array)

    array_shape_x, array_shape_y = array.shape

    kernel_shape_x, kernel_shape_y = kernel_shape

    for index_array_x in numba.prange(array_shape_x):
        for index_array_y in numba.prange(array_shape_y):

            values = np.empty(kernel_shape)

            for index_kernel_x in numba.prange(kernel_shape_x):
                for index_kernel_y in numba.prange(kernel_shape_y):

                    position_kernel_x = index_kernel_x - kernel_shape_x // 2
                    position_kernel_y = index_kernel_y - kernel_shape_y // 2

                    index_final_x = index_array_x + position_kernel_x
                    index_final_y = index_array_y + position_kernel_y

                    index_final_x = np.abs(index_final_x)
                    index_array_y = np.abs(index_array_y)

                    if index_final_x >= array_shape_x:
                        index_final_x = ~(index_final_x % array_shape_x)

                    if index_final_y >= array_shape_y:
                        index_final_y = ~(index_final_y % array_shape_y)

                    values[index_kernel_x, index_kernel_y] = array[index_final_x, index_final_y]

            thresh_min = np.percentile(values, proportion)
            thresh_max = np.percentile(values, 1 - proportion)

            sum_convol = 0
            num_convol = 0
            for index_kernel_x in numba.prange(kernel_shape_x):
                for index_kernel_y in numba.prange(kernel_shape_y):
                    value = values[index_kernel_x, index_kernel_y]
                    if thresh_min < value < thresh_max:
                        sum_convol += value
                        num_convol += 1

            result[index_array_x, index_array_y] = sum_convol / num_convol

    return result


@numba.njit(parallel=True,)
def _mean_trimmed_numba_3d(
        array: np.ndarray,
        kernel_shape: tuple[int, int, int] = (11, 11, 11),
        proportion: float = 0.25,
):
    result = np.empty_like(array)

    array_shape_x, array_shape_y, array_shape_z = array.shape

    kernel_shape_x, kernel_shape_y, kernel_shape_z = kernel_shape

    for index_array_x in numba.prange(array_shape_x):
        for index_array_y in numba.prange(array_shape_y):
            for index_array_z in numba.prange(array_shape_z):

                values = np.empty(kernel_shape)

                for index_kernel_x in numba.prange(kernel_shape_x):
                    for index_kernel_y in numba.prange(kernel_shape_y):
                        for index_kernel_z in numba.prange(kernel_shape_z):

                            position_kernel_x = index_kernel_x - kernel_shape_x // 2
                            position_kernel_y = index_kernel_y - kernel_shape_y // 2
                            position_kernel_z = index_kernel_z - kernel_shape_z // 2

                            index_final_x = index_array_x + position_kernel_x
                            index_final_y = index_array_y + position_kernel_y
                            index_final_z = index_array_z + position_kernel_z

                            index_final_x = np.abs(index_final_x)
                            index_array_y = np.abs(index_array_y)
                            index_array_z = np.abs(index_array_z)

                            if index_final_x >= array_shape_x:
                                index_final_x = ~(index_final_x % array_shape_x)

                            if index_final_y >= array_shape_y:
                                index_final_y = ~(index_final_y % array_shape_y)

                            if index_final_z >= array_shape_z:
                                index_final_z = ~(index_final_z % array_shape_z)

                            values[index_kernel_x, index_kernel_y, index_kernel_z] = array[index_final_x, index_final_y, index_final_z]

                thresh_min = np.percentile(values, proportion)
                thresh_max = np.percentile(values, 1 - proportion)

                sum_convol = 0
                num_convol = 0
                for index_kernel_x in numba.prange(kernel_shape_x):
                    for index_kernel_y in numba.prange(kernel_shape_y):
                        for index_kernel_z in numba.prange(kernel_shape_z):
                            value = values[index_kernel_x, index_kernel_y, index_kernel_z]
                            if thresh_min < value < thresh_max:
                                sum_convol += value
                                num_convol += 1

                result[index_array_x, index_array_y, index_array_z] = sum_convol / num_convol

    return result

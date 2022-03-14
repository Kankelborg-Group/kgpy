import numpy as np
import scipy.signal
from astropy import units as u
import astropy.convolution


def calc_kernel(
        ndims: int = 4,
        x_axis_index: int = ~2,
        y_axis_index: int = ~1,
        base_kernel: u.Quantity = u.Quantity([0.25, 0.5, 0.25])
) -> u.Quantity:
    sh = [1, ] * ndims
    sh[x_axis_index] = 3
    sh[y_axis_index] = 3
    p = np.zeros(sh)
    k = base_kernel
    k = np.expand_dims(k, ~0) * k
    k /= k.sum()
    sl = [0] * ndims
    sl[x_axis_index] = slice(None)
    sl[y_axis_index] = slice(None)
    p[tuple(sl)] = k
    return p


def apply(
        data: np.ndarray,
        x_axis_index: int = ~2,
        y_axis_index: int = ~1,
        user_provided_kernel: bool = False,
        kernel: u.Quantity = u.Quantity([0.25, 0.5, 0.25]),
) -> np.ndarray:
    """
    Apply the antialias kernel to the cube `data`, for use in MART related inversion problems.
    :param data:
    :param x_axis_index: axis in `data` that is the spatial x-axis
    :param y_axis_index: axis in `data` that is the spatial y-axis
    :param user_provided_kernel: if True, do not use `calc_kernel` to calculate the convolution kernel, instead using a
    kernel provided by the user.
    :param kernel: 1-dimensional kernel to be given to `calc_kernel` to generate the convolution kernel, or, if
    `user_provided_kernel` True, this kernel is handed directly to the convolution
    :return: antialiased version of `data`
    """
    if user_provided_kernel:
        aa_kernel = kernel
    else:
        aa_kernel = calc_kernel(
            ndims=data.ndim,
            x_axis_index=x_axis_index,
            y_axis_index=y_axis_index,
            base_kernel=kernel)

    data = scipy.signal.convolve(data, aa_kernel, mode='same', method='direct')

    return data

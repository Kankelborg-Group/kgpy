import numpy as np
import scipy

from . import freq


def periodogram_fft(
        data: np.ndarray,
        retain_dc: bool = False
):
    """
    This is a port of CCK's `pperiodogram_fft.pro` to Python 3. Ported by NCG, 2020-07-30

    Perform the simplest reasonable power spectral estimate, using the FFT and a Hann window. The normalization is
    such that the mean value of the result is approximately the mean squared value of the data. Works for 1D or 2D
    inputs. It should be noted that the expected error of this estimate of the power spectrum is about 100% (see
    Numerical Recipes). However, this routine is needed by the much better estimators powerspec and spec2d.

    :param data: 1d or 2d array for which the power spectrum is desired. It is not assumed that the data is periodic;
    therefore, the data will be Hann windowed. It is also not assumed that the image is real. That's why the
    normalization is such that the sum must be performed over both positive and negative k.
    :param retain_dc: if True, then retain the DC offset when calculating the FFT. Ordinarily, the DC offset is
    subtracted before calculating the spectrum, because the Hanning window tends to make the offset bleed into the
    nearest neighbors of the DC element of the FFT.
    :return:
    """
    if not retain_dc:
        data = data - np.mean(data)

    ndim = data.ndim
    if ndim == 2:
        # the case where data is an image
        nx, ny = data.shape
        hann = np.outer(scipy.signal.windows.hann(nx), scipy.signal.windows.hann(ny))
        power = np.abs(scipy.fft.fftn(data * hann)) ** 2 * 7.1111111 * nx * ny
        # the factor 7.1111111 is one over the mean square of the Hann window
    elif ndim == 1:
        nx = data.shape
        power = np.abs(scipy.fft.fftn(data * scipy.signal.windows.hann(nx))) ** 2 * 2.6666667 * nx
        # the factor 2.6666667 is one over the mean square of the Hann window
    else:
        raise ValueError('Data must be a 1D or 2D array.')

    return power


def spec(
        data: np.ndarray,
        n_frequencies: int = 16,
        dx: float = 1.0,
        radians: bool = False,
        retain_dc: bool = False,
):
    """
    This is a port of CCK's `powerspec.pro` to Python 3. Ported by NCG, 2020-08-02

    Compute power spectrum of a 1D array by dividing into multiple overlapping segments and averaging the power
    spectra. This improves the estimate at the expense of spectral resolution. The normalization convention is the
    same as with periodogram_fft.pro. The method is a particular implementation of Welch's method (IEEE Trans. Audio
    & Electroacoustics, 15:2:70-73, June, 1967). I have used the Hanning cosinusoidal window.

    :param data:
    :param n_frequencies:
    :param dx:
    :param radians:
    :param retain_dc:
    :return:
    """
    if data.ndim != 1:
        raise ValueError('Argument `data` is expected to be 1-dimensional')

    if n_frequencies < 4:
        raise Warning('Minimum n_frequencies is 4, and has been set to 4')
        n_frequencies = 4

    n_segments = np.ceil(2 * data.size / n_frequencies) - 1
    # How many segments of length n_frequencies fit in the interval, if we overlap them like 2 courses of bricks?

    if n_segments < 2:
        raise ValueError('Data array too short to computer power spectrum using specified segment size')

    xsep = (data.size - n_frequencies) / (n_segments - 1)
    spectra = np.zeros((n_segments, n_frequencies))
    for i in range(n_segments):
        x0 = int(np.floor(i * xsep))
        x1 = x0 + n_frequencies
        segment = data[x0:x1]
        spectra[i, :] = periodogram_fft(segment, retain_dc=retain_dc)

    power = np.nanmean(spectra, axis=0)

    k = freq.k_arr(n=n_frequencies, dx=dx, radians=radians)

    return power, k


def spec2d(
        data: np.ndarray,
        sub_image_size: int = 16,
        retain_dc: bool = False,
):
    """
    This is a port of CCK's `spec2d.pro` to Python 3. Ported to Python by NCG, 2020-07-30

    Compute the power spectra of a 2D array by dividing into many overlapping sub-images and averaging their power
    spectra. This impproves the estiate at the expense of spectral resolution. This is a 2d analog of `powerspec`, which
    uses Welch's method with a Hanning window. Any sub-image containing NaN is ignored by `np.nanmean`

    :param data: 2D array of data
    :param sub_image_size: size of the square sub-image to analyze. Default is 16, min is 4
    :param retain_dc: if True, then retain the DC offset when calculating the power spectrum. Ordinarily, the DC offset
    is subtracted before calculating the spectrum, because the Hanning window tends to make the offset bleed into the
    nearest neighbor of pixel (0, 0) of the 2D power spectrum.
    :return:
    """

    nx, ny = data.shape

    if sub_image_size < 4:
        # force the sub_image_size to be at least 4
        sub_image_size = 4

    # how many segments of length sub_image_size fit into the intercal, if we overlap them like 2 courses of bricks.
    number_x_segments = int(np.ceil(2 * nx / sub_image_size) - 1)
    number_y_segments = int(np.ceil(2 * ny / sub_image_size) - 1)

    if number_x_segments < 2:
        raise ValueError('Data array too narrow to compute power spectrum using specified segment size')
    if number_y_segments < 2:
        raise ValueError('Data array too short to compute power spectrum using specified segment size')

    x_seperation = (nx - sub_image_size) / (number_x_segments - 1) * (1 - 1e-6)
    y_seperation = (ny - sub_image_size) / (number_y_segments - 1) * (1 - 1e-6)
    #   the last factor here is to ensure we can fit the correct number of segments within the data array

    spectra = np.zeros((number_x_segments, number_y_segments, sub_image_size, sub_image_size))
    for i in range(number_x_segments):
        x0 = int(np.floor(i * x_seperation))
        x1 = x0 + sub_image_size
        for j in range(number_y_segments):
            y0 = int(np.floor(j * y_seperation))
            y1 = y0 + sub_image_size
            segment = data[x0:x1, y0:y1]
            spectra[i, j, :, :] = periodogram_fft(segment, retain_dc=retain_dc)

    power = np.nanmean(spectra, axis=(0, 1))

    return power


def spec_xy(
        image: np.ndarray,
        nfreq: int = 64,
        dx: float = 1.0,
        radians: bool = False,
        retain_dc: bool = False
):
    """
    Estimate horizontal and vertical power spectra of an image. The horizontal power spectrum, for example,
    is calculated as an average of the power spectra of the rows. Moreover, the power spectrum of each row is
    estimated by averaging over a sequence of short segments along the row. Columns are treated similarly. The
    normalization convention is the same as with `periodogram_fft`

    :param image: the 2d image for which the power spectra are to be calculated :param nfreq: Number of elements in
    each segment, and also the number of elements in each of the output power spectra. The smaller the `nfreq`,
    thee lower the spectral resolution but the better the noise supression.
    :param dx: sample interval. Defaults to 1. If included, then `k` will be converted to units of cycles (or radians,
    if that keyword is set) per unit distance (or time), corresponding to the units of dx
    :param radians: if True, then let `k` be in radians, rather than cycles, per whatever.
    :param retain_dc: if set, then retain the DC offset when calculating the power spectrum. Ordinarily, the DC offset
    is subtracted before calculating the spectrum, because the Hanning window tends to make the offset bleed into the
    nearest neighbors of the pixel [0,0] of the 2D power spectrum.
    :return: `psx`, `psy`: the power spectra averages of `x` and `y`, respectively
    """

    nx, ny = image.shape

    psx = np.zeros(nfreq)
    n_valid = ny
    for y in range(ny):
        ps = spec(image[:, y], nfreq, dx=dx, radians=radians, retain_dc=retain_dc)
        if np.sum(np.logical_or(np.isnan(ps), np.isinf(ps))) != 0:
            n_valid -= 1  # one less than we thought
        else:
            psx += ps
    psx = psx / n_valid

    # Now, for the other dimension
    n_valid =  nx
    psy = np.zeros(nfreq)
    for x in range(nx):
        ps = spec(image[x, :], nfreq, dx=dx, radians=radians, retain_dc=retain_dc)
        if np.sum(np.logical_or(np.isnan(ps), np.isinf(ps))) != 0:
            n_valid -= 1  # one less than we thought
        else:
            psy += ps
    psy = psy / n_valid

    return psx, psy


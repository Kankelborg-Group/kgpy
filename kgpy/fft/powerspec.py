import numpy as np
import scipy
import scipy.signal

__all__ = [
    'k_arr',
    'k_arr2d',
    'periodogram_fft',
    'powerspec2d',
    'random_image'
]


def k_arr(
        n: int,
        dx: float = 1.0,
        radians: bool = False
):
    """
    This is a port of CCK's `k_arr.pro` to Python 3. Ported by NCG, 2020-07-31

    Generate an array of wave-number values (by default in units of cycles per sample interval), arranged corresponding
    to the FFT of an N-element array.

    **Note**: This is similar, but not identical, to using the helper function `scipy.fft.fftfreq` to generate arrays of
    frequencies. The main difference is that `scipy.fft.fftfreq` is limited to one dimension, and does not have the
    keyword argument to convert between radians snd cycles. -NCG

    :param n: number of elements
    :param dx: sample interval. If included, then k will be converted into units of
    cycles (or radians, if that keyword is set) per unit distance (or time), corresponding to the units of dx.
    :param radians: if True, then let `k` be in radians, instead of the default cycles
    :return: array `k`, `n` element array of positive and negative wave-numbers, where the maximum (Nyquist) frequency
    has a value of pi radians per sample interval. The `dx` and `radians` keywords can be used to modidy the units of
    `k`.
    """
    dk = 1.0 / (n * dx)  # frequency interval equals the fundamental frequency of one cycle per n samples,
    # or one cycle per total distance of n*dx
    positives = np.arange(np.floor((n / 2)))
    negatives = -1 * np.arange(np.ceil(n / 2))[::-1]
    k = dk * np.hstack((positives, negatives))
    if radians:
        k *= 2 * np.pi
    return k


def k_arr2d(
        nx: int,
        ny: int,
        dx: float = 1.0,
        dy: float = 1.0,
        radians: bool = False,
):
    """
    This is a port of CCK's `k_arr2d.pro` to Python 3. Ported by NCG, 2020-07-30

    Generate an array of wavenumber values (by default, in units of cycles per
    sample interval), arranged corresponding to the FFT of an `nx` by `ny`-element array.

    **Note**: This is similar, but not identical, to using the helper function `scipy.fft.fftfreq` twice to generate
    arrays of frequencies. The main difference is that `scipy.fft.fftfreq` is limited to one dimension, and does not
    have the keyword argument to convert between radians snd cycles. -NCG

    :param nx: size of array in x-dimension (axis 0)
    :param ny: size of array in y-dimension (axis 1)
    :param dx: sample interval. If included, `k` will be converted to units of cycles (or radians, see `radian` keyword)
    per unit distance (or time), corresponding to the units of dx.
    :param dy: sample interval. If included, `k` will be converted to units of cycles (or radians, see `radian` keyword)
    per unit distance (or time), corresponding to the units of dy.
    :param radians:
    :return: `k`: `kx`, an array of horizontal wave numbers; `ky`: an array of vertical wavenumbers; big_kx, big_ky:
    `nx` by `ny` arrays of horizontal vertical wavenumbers, a meshgrid.
    """
    kx = k_arr(nx, dx, radians=radians)
    ky = k_arr(ny, dy, radians=radians)
    big_kx, big_ky = np.meshgrid(kx, ky)
    k = np.sqrt(big_kx ** 2 + big_ky ** 2)

    return k, kx, ky, big_kx, big_ky


def periodogram_fft(
        data: np.ndarray,
        retain_dc: bool = False
):
    """
    This is a port of CCK's `pperiodogram_fft.pro` to Python 3. Ported by NCG, 2020-07-30

    Perform the simplest reasonable power spectral estimate, using the FFT and a Hann window. The normalization is
    such that the mean value of the result is approximately the mean squared value of the data. Works for 1D or 2D
    inputs. It should be noted that the expected error of this estimate of the power spectrum is about 100% (see
    Numerical Recipes). However, this routine is needed by the much better estimators powerspec and powerspec2d.

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


def powerspec2d(
        data: np.ndarray,
        sub_image_size: int = 16,
        retain_dc: bool = False,
):
    """
    This is a port of CCK's `powerspec2d.pro` to Python 3. Ported to Python by NCG, 2020-07-30

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


def random_image(
        nx: int,
        ny: int,
        alpha: float,
        meanval: float,
        pad: int = 1,
        keep_complex: bool = False
):
    """
    Port of CCK's `random_image.pro` to Python. Ported by NCG 2020-07-30

    Generate a random image with a specified power law spectrum. The result is a 2D array with all positive values. The
    result will be periodic unless padding is used.

    **Note**: In CCK's orginal IDL version of this function, a real-valued array was returned without needing to take an
    abs() of the final image. Here, I have added a keyword to let the user decide between returning a complex-valued
    array, or the abs() of that complex valued array, depending on the user's desire for phase information. For most
    practical applications, I imagine that only the amplitude will be needed. -NCG

    :param Nx: size of image in x direction
    :param Ny: size of image in y direction
    :param alpha: power law index for the image power spectrum (goes as k^(-alpha))
    :param meanval: mean value of the image.
    :param pad: positive integer factor for padding arrays. pad=2 or higher prevents the output image from being
    periodic in structure.
    :return:
    """

    k, _, _, _, _ = k_arr2d(nx * pad, ny * pad, )
    powerspec = k ** (-alpha)
    powerspec[0, 0] = 0.0

    image = np.random.random((nx * pad, ny * pad))
    image_fft = scipy.fft.fftn(image)
    image_power = np.abs(image_fft) ** 2
    image_fft *= np.sqrt(powerspec / image_power)  # might be better not to have the 'image_power' in the denominator,
    # for better stochastic process

    image = scipy.fft.ifftn(image_fft)[0:nx, 0:ny]  # removes padding
    image -= np.min(image)
    image *= meanval / np.mean(image)

    if not keep_complex:
        image = np.abs(image)

    return image

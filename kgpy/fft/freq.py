import numpy as np


def k_arr(
        n: int,
        dx: float = 1.0,
        radians: bool = False
):
    """This is a port of CCK's `k_arr.pro` to Python 3. Ported by NCG, 2020-07-31

    Generate an array of wave-number values (by default in units of cycles per sample interval), arranged corresponding
    to the FFT of an N-element array.
    This is similar, but not identical, to using the helper function `scipy.fft.fftfreq` to generate arrays of
    frequencies. The main difference is that `scipy.fft.fftfreq` is limited to one dimension, and does not have the
    keyword argument to convert between radians snd cycles. -NCG

    :param n: number of elements
    :param dx: sample interval. If included, then k will be converted into units of cycles (or radians, if that keyword
        is set) per unit distance (or time), corresponding to the units of dx.
    :param radians: if True, then let `k` be in radians, instead of the default cycles
    :return: array `k`, `n` element array of positive and negative wave-numbers, where the maximum (Nyquist) frequency
        has a value of pi radians per sample interval. The `dx` and `radians` keywords can be used to modidy the units
        of `k`.
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
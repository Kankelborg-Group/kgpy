import numpy as np
import scipy
import scipy.signal

__all__ = [
    'power_law_image'
]

from kgpy.fft import k_arr2d


def power_law_image(
        nx: int,
        ny: int,
        alpha: float,
        meanval: float,
        pad: int = 1,
        keep_complex: bool = False
):
    """
    Port of CCK's `power_law_image.pro` to Python. Ported by NCG 2020-07-30

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

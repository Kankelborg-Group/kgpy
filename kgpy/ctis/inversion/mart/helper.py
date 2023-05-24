import numpy as np
import pathlib as pl
import typing as typ
import astropy.units as u
import astropy.io.fits
import astropy.wcs
import dataclasses
import kgpy.ctis.inversion
from . import mart
from kgpy.observatories.iris import mosaics

__all__ = ['good_mosaics', 'image_setup', 'add_poisson_noise', 'generate_projections', 'Moments', 'result_to_moments']


def good_mosaics():
    """
    I found that there are a number of mosaics in the "mosaics.download()" function that are not 6000px wide, and so
    they are not useful for the tests I am doing, as the image_setup below requires 6000px. This 6000px requirement was
    arbitrarily chosen when we started designing and testing MART on IRIS data, and its sorta become the defacto for how
    I'm testing MART.

    This funciton will return a list of the locations (as pathlib.Path objects) of the good IRIS fits files, as well as
    list of indices of these "good" files within the originally downloaded list, as that might be useful
    :return:
    """
    all_fits_files = mosaics.download()
    fits_files_directory = all_fits_files[0].parent
    bad_fits_files = [
        'IRISMosaic_20141014_Si1393.fits.gz',
        'IRISMosaic_20141020_Si1393.fits.gz',
        'IRISMosaic_20150301_Si1393.fits.gz',
        'IRISMosaic_20151012_Si1393.fits.gz',
        'IRISMosaic_20151018_Si1393.fits.gz',
        'IRISMosaic_20151027_Si1393.fits.gz',
        'IRISMosaic_20160501_Si1393.fits.gz',
        'IRISMosaic_20161017_Si1393.fits.gz',
        'IRISMosaic_20170312_Si1393.fits.gz',
        'IRISMosaic_20180312_Si1393.fits.gz',
        'IRISMosaic_20180422_Si1393.fits.gz',
    ]

    bad_fits_files = [fits_files_directory / pl.Path(f) for f in bad_fits_files]

    good_fits_files = []
    idx = []
    for j, fn in enumerate(all_fits_files):
        if fn not in bad_fits_files:
            good_fits_files.append(fn)
            idx.append(j)

    return good_fits_files, idx


def image_setup(
        fits_file: pl.Path,
        x_range: typ.Tuple[int, int],
        y_range: typ.Tuple[int, int],
        saa_x_ranges: typ.List[typ.Tuple[int, int]],
        saa_y_ranges: typ.List[typ.Tuple[int, int]],
        border: int = 20,
        rebin_factor: int = 6,
        wavelength_rebin_factor: int = 2,
        roy_smart_rebin_error: int = 6,
        cval=1

) -> typ.Tuple[np.ndarray, np.ndarray]:
    """
    Sets up an image for use in MART from a given FITS file. Assumes image has a WCS and is a spatial (xy) and spectral
    image (in last axis). This is a very brittle program.

    :param roy_rebin_error: Roy, when writing down the SAA regions, apparently introduced a factor of 6 error.
        Incoming y-values for SAA regions needed to be multiplied by 6 to be correctly applied to the raw data (not
        reshaped or rebinned) from the FITS file.
    :param fits_file: fits file to use
    :param x_range: (x_min, x_max) range to be used in the image.
    :param y_range: (y_min, y_max) range to be used in the image.
    :param border: How much cushion for rotations?
    :return:
    """

    hdu = astropy.io.fits.open(fits_file)[0]
    header = hdu.header

    wcs = astropy.wcs.WCS(header)
    wcs = wcs.swapaxes(0, -1)
    wcs = wcs.swapaxes(-1, -2)

    data = hdu.data

    # Zero out SAA regions
    for idx in range(len(saa_x_ranges)):
        saa_x0 = saa_x_ranges[idx][0]
        saa_x1 = saa_x_ranges[idx][1]
        saa_y0 = roy_smart_rebin_error * saa_y_ranges[idx][0]
        saa_y1 = roy_smart_rebin_error * saa_y_ranges[idx][1]
        data[:, saa_y0:saa_y1, saa_x0:saa_x1] = cval

    data = data.transpose()
    data = data[:1000, :6000, :40]
    sh = data.shape

    # Especially confused about these next three lines:
    data = data.reshape(
        (sh[0], sh[1] // rebin_factor, rebin_factor, sh[2] // wavelength_rebin_factor, wavelength_rebin_factor))
    data = data.sum((2, 4))
    data[data < 0] = 0

    region = data[x_range[0]:x_range[1], y_range[0]:y_range[1]]

    # Add a border
    region = np.pad(region, [(border, border), (border, border), (0, 0)])

    return region, wcs


def add_poisson_noise(
        image: np.ndarray
):
    """
    Given a data array, add poisson type noise
    :param image:
    :return: image with poisson noise added
    """
    noise = np.random.poisson(image).astype(np.float)
    return noise


default_rotation_kwargs = {'reshape': False, 'prefilter': False, 'order': 3, 'mode': 'nearest', }


def generate_projections(
        data: np.ndarray,
        angles: u.Quantity,
        spectral_order: int = 1,
        poisson_noise: bool = False,
        rotation_kwargs: typ.Dict = default_rotation_kwargs,
        projection_shape=None
) -> np.ndarray:
    """
    Given a data cube a list of angles, create a projection through the cube for each of those angles at the given
    spectral order.
    :param poisson_noise: if True, add poisson noise to the projection
    :param data:
    :param angles: array of angles, must be astropy.units.Quantity, as the forward model takes this as an input
    :param spectral_order:
    :return:
    """

    projections = []

    for angle in angles:
        proj = kgpy.ctis.inversion.mart.forward.model(
            cube=data,
            projection_azimuth=angle,
            spectral_order=spectral_order,
            rotation_kwargs=rotation_kwargs,
            projection_shape=projection_shape
        )
        if poisson_noise:
            proj[proj <= 0] = 0
            proj = add_poisson_noise(proj)
        projections.append(proj)

    return np.array([projections])


@dataclasses.dataclass
class Moments:
    intensity: np.ndarray or list
    shift: u.Quantity or np.ndarray or list
    width: u.Quantity or np.ndarray or list
    skew: u.Quantity or np.ndarray or list
    metadata: typ.Dict[str, typ.Any]
    """
    A class to basically just keep moments and the correct MART parameters associated with them together. 
    """


def result_to_moments(
        recovered: mart.Result,
        base_wavelength: u.Quantity,
        wcs: astropy.wcs.WCS,
        wscale: int = 1,
        axis: int = ~0,
):
    intensity, shift, width, skew = mosaics.line_profile_moments.first_four_moments(
        cube=recovered.cube,
        base_wavelength=base_wavelength,
        wcs=wcs,
        wscale=wscale,
        axis=axis
    )
    meta = {
        'anti_aliasing': recovered.object_parameters['anti_aliasing'],
        'rotation_kwargs': recovered.object_parameters['rotation_kwargs']
    }

    return Moments(intensity, shift, width, skew, meta)

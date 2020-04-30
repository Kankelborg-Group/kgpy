import typing as typ
import numpy as np
import astropy.units as u
import astropy.constants
import astropy.wcs
from kgpy.moment.percentile import arg_percentile, intensity

__all__ = ['indices_to_velocity', 'velocity_arg_percentile', 'shift', 'width', 'skew', 'first_four_moments']


def indices_to_velocity(
        indices: np.ndarray,
        base_wavelength: u.Quantity,
        wcs: astropy.wcs.WCS,
        wscale: int
):
    wavelengths, _, _ = wcs.all_pix2world(wscale * indices, 0, 0, 0)
    wavelengths = wavelengths << u.AA
    velocities = astropy.constants.c * (wavelengths - base_wavelength) / base_wavelength
    velocities = velocities.to(u.km / u.s)

    return velocities


def velocity_arg_percentile(
        cube: np.ndarray,
        percentile: float,
        base_wavelength: u.Quantity,
        wcs: astropy.wcs.WCS,
        wscale: int = 1,
        axis: int = ~0,
) -> np.ndarray:
    return indices_to_velocity(
        indices=arg_percentile(cube, percentile, axis=axis),
        base_wavelength=base_wavelength,
        wcs=wcs,
        wscale=wscale
    )

def shift(
        cube: np.ndarray,
        base_wavelength: u.Quantity,
        wcs: astropy.wcs.WCS,
        wscale: int = 1,
        axis: int = ~0,
) -> np.ndarray:
    return velocity_arg_percentile(cube, 0.5, base_wavelength, wcs, wscale, axis)


def width(
        cube: np.ndarray,
        base_wavelength: u.Quantity,
        wcs: astropy.wcs.WCS,
        wscale: int = 1,
        axis: int = ~0,
) -> np.ndarray:
    p1 = velocity_arg_percentile(cube, 0.25, base_wavelength, wcs, wscale, axis)
    p3 = velocity_arg_percentile(cube, 0.75, base_wavelength, wcs, wscale, axis)
    return p3 - p1


def skew(
        cube: np.ndarray,
        base_wavelength: u.Quantity,
        wcs: astropy.wcs.WCS,
        wscale: int = 1,
        axis: int = ~0,
) -> np.ndarray:
    p1 = velocity_arg_percentile(cube, 0.25, base_wavelength, wcs, wscale, axis)
    p2 = velocity_arg_percentile(cube, 0.50, base_wavelength, wcs, wscale, axis)
    p3 = velocity_arg_percentile(cube, 0.75, base_wavelength, wcs, wscale, axis)
    return p3 - 2 * p2 + p1


def first_four_moments(
        cube: np.ndarray,
        base_wavelength: u.Quantity,
        wcs: astropy.wcs.WCS,
        wscale: int = 1,
        axis: int = ~0,
) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I0 = intensity(cube, axis=axis)
    I1 = shift(cube, base_wavelength, wcs, wscale, axis)
    I2 = width(cube, base_wavelength, wcs, wscale, axis)
    I3 = skew(cube, base_wavelength, wcs, wscale, axis)

    I0 = I0.flatten()
    I1 = I1.flatten()
    I2 = I2.flatten()
    I3 = I3.flatten()

    return I0, I1, I2, I3

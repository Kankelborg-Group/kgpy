import pathlib
import astropy.units as u
import kgpy.chianti

__all__ = ['angular_radius_max']

# https://en.wikipedia.org/wiki/Angular_diameter#Use_in_astronomy
angular_radius_max = (32 * u.arcmin + 32 * u.arcsec) / 2  # type: u.Quantity
"""maximum angular radius of the solar disk"""

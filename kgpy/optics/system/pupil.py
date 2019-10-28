import typing as tp
import astropy.units as u

from kgpy import optics

__all__ = ['normalize_coordinates', 'normalize_pupil_coordinates_inverse']


# noinspection PyPep8Naming
def normalize_coordinates(pupil_x: u.Quantity,
                          pupil_y: u.Quantity,
                          entrance_pupil_radius: u.Quantity
                          ) -> tp.Tuple[u.Quantity, u.Quantity]:
    pupil_norm_x = pupil_x / entrance_pupil_radius
    pupil_norm_y = pupil_y / entrance_pupil_radius

    return pupil_norm_x, pupil_norm_y


# noinspection PyPep8Naming
def normalize_pupil_coordinates_inverse(pupil_norm_x: u.Quantity,
                                        pupil_norm_y: u.Quantity,
                                        entrance_pupil_radius: u.Quantity
                                        ) -> tp.Tuple[u.Quantity, u.Quantity]:
    p_x = pupil_norm_x * entrance_pupil_radius
    p_y = pupil_norm_y * entrance_pupil_radius

    return p_x, p_y



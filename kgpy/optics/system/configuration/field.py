import typing as tp
import enum
import numpy as np
import astropy.units as u

from kgpy import optics

__all__ = ['Field']


class Field:

    def __init__(self,
                 x: u.Quantity = None,
                 y: u.Quantity = None,
                 weight: u.Quantity = None,
                 vdx: u.Quantity = None,
                 vdy: u.Quantity = None,
                 vcx: u.Quantity = None,
                 vcy: u.Quantity = None,
                 van: u.Quantity = None
                 ):

        if x is None:
            x = 0.0 * u.rad
        if y is None:
            y = 0.0 * u.rad
        if weight is None:
            weight = 1.0 * u.dimensionless_unscaled
        if vdx is None:
            vdx = 0.0 * u.dimensionless_unscaled
        if vdy is None:
            vdy = 0.0 * u.dimensionless_unscaled
        if vcx is None:
            vcx = 0.0 * u.dimensionless_unscaled
        if vcy is None:
            vcy = 0.0 * u.dimensionless_unscaled
        if van is None:
            van = 0.0 * u.rad

        self._x = x
        self._y = y
        self._weight = weight
        self._vdx = vdx
        self._vdy = vdy
        self._vcx = vcx
        self._vcy = vcy
        self._van = van

    @classmethod
    def from_physical_pupil_size(cls,
                                 field_x: u.Quantity,
                                 field_y: u.Quantity,
                                 pupil_decenter_x: u.Quantity,
                                 pupil_decenter_y: u.Quantity,
                                 pupil_semi_axis_x: u.Quantity,
                                 pupil_semi_axis_y: u.Quantity,
                                 entrance_pupil_radius: u.Quantity,
                                 ) -> 'Field':

        vdx, vdy = optics.system.configuration.pupil.normalize_coordinates(pupil_decenter_x,
                                                                           pupil_decenter_y,
                                                                           entrance_pupil_radius)

        vcx = 1 - (pupil_semi_axis_x / entrance_pupil_radius)
        vcy = 1 - (pupil_semi_axis_y / entrance_pupil_radius)

        print(vcy)

        weight = 1.0 * u.dimensionless_unscaled
        van = 0 * u.deg

        return cls(field_x, field_y, weight, vdx, vdy, vcx, vcy, van)

    @property
    def x(self) -> u.Quantity:
        return self._x

    @property
    def y(self) -> u.Quantity:
        return self._y

    @property
    def weight(self) -> u.Quantity:
        return self._weight

    @property
    def vdx(self) -> u.Quantity:
        return self._vdx

    @property
    def vdy(self) -> u.Quantity:
        return self._vdy

    @property
    def vcx(self) -> u.Quantity:
        return self._vcx

    @property
    def vcy(self) -> u.Quantity:
        return self._vcy

    @property
    def van(self) -> u.Quantity:
        return self._van

    @property
    def radial_magnitude(self):
        return np.sqrt(self.x * self.x + self.y * self.y)

    def vignetting_factors(self,
                           pupil_norm_x: u.Quantity,
                           pupil_norm_y: u.Quantity,
                           ) -> tp.Tuple[u.Quantity, u.Quantity]:

        P_x = pupil_norm_x
        P_y = pupil_norm_y

        P_x_ = self.vdx + P_x * (1 - self.vcx)
        P_y_ = self.vdy + P_y * (1 - self.vcy)

        P_x__ = P_x_ * np.cos(self.van) - P_y_ * np.sin(self.van)
        P_y__ = P_x_ * np.sin(self.van) + P_y_ * np.cos(self.van)

        return P_x__, P_y__


def max_field_magnitude_radial(fields: tp.List[Field]) -> u.Quantity:
    return max(field.radial_magnitude for field in fields)


def max_field_magnitude_rectangular(fields: tp.List[Field]) -> tp.Tuple[u.Quantity, u.Quantity]:
    max_x = max(field.x for field in fields)
    max_y = max(field.y for field in fields)

    return max_x, max_y


# noinspection PyPep8Naming
def normalization_factor(max_field: tp.Union[u.Quantity, tp.Tuple[u.Quantity, u.Quantity]]
                         ) -> tp.Tuple[u.Quantity, u.Quantity]:
    try:

        F_x = max_field[0]
        F_y = max_field[1]

    except TypeError:

        F_x = max_field
        F_y = max_field

    return F_x, F_y


# noinspection PyPep8Naming
def normalized_field_coordinate_transform(f_x: u.Quantity,
                                          f_y: u.Quantity,
                                          max_field: tp.Union[u.Quantity, tp.Tuple[u.Quantity, u.Quantity]]
                                          ) -> tp.Tuple[u.Quantity, u.Quantity]:
    F_x, F_y = normalization_factor(max_field)

    H_x = f_x / F_x
    H_y = f_y / F_y

    return H_x, H_y


# noinspection PyPep8Naming
def normalized_field_coordinate_inverse_transform(H_x: u.Quantity,
                                                  H_y: u.Quantity,
                                                  max_field: tp.Union[u.Quantity, tp.Tuple[u.Quantity, u.Quantity]]
                                                  ) -> tp.Tuple[u.Quantity, u.Quantity]:
    F_x, F_y = normalization_factor(max_field)

    f_x = H_x * F_x
    f_y = H_y * F_y

    return f_x, f_y

import dataclasses
import typing as tp
import numpy as np
import astropy.units as u

from . import pupil

__all__ = ['Fields']


@dataclasses.dataclass
class Fields:

    x: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.rad)
    y: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.rad)
    weight: u.Quantity = dataclasses.field(default_factory=lambda: [1.0] * u.dimensionless_unscaled)
    vdx: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.dimensionless_unscaled)
    vdy: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.dimensionless_unscaled)
    vcx: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.dimensionless_unscaled)
    vcy: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.dimensionless_unscaled)
    van: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.rad)

    @property
    def config_broadcast(self):
        return np.broadcast(
            self.x[..., 0],
            self.y[..., 0],
            self.weight[..., 0],
            self.vdx[..., 0],
            self.vdy[..., 0],
            self.vcx[..., 0],
            self.vcy[..., 0],
            self.van[..., 0],
        )
    
    @property
    def num_per_config(self):
        b = np.broadcast(self.x, self.y, self.weight, self.vdx, self.vdy, self.vcx, self.vcy, self.van)
        return b.shape[~0]


    @classmethod
    def from_physical_pupil_size(cls,
                                 field_x: u.Quantity,
                                 field_y: u.Quantity,
                                 pupil_decenter_x: u.Quantity,
                                 pupil_decenter_y: u.Quantity,
                                 pupil_semi_axis_x: u.Quantity,
                                 pupil_semi_axis_y: u.Quantity,
                                 entrance_pupil_radius: u.Quantity,
                                 ) -> 'Fields':
        vdx, vdy = pupil.normalize_coordinates(pupil_decenter_x,
                                               pupil_decenter_y,
                                               entrance_pupil_radius)

        vcx = 1 - (pupil_semi_axis_x / entrance_pupil_radius)
        vcy = 1 - (pupil_semi_axis_y / entrance_pupil_radius)

        weight = [1.0] * u.dimensionless_unscaled
        van = [0] * u.deg

        return cls(field_x, field_y, weight, vdx, vdy, vcx, vcy, van)

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


def max_field_magnitude_radial(fields: tp.List[Fields]) -> u.Quantity:
    return max(field.radial_magnitude for field in fields)


def max_field_magnitude_rectangular(fields: tp.List[Fields]) -> tp.Tuple[u.Quantity, u.Quantity]:
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

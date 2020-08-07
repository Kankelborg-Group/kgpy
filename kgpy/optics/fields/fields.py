import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

from .. import pupil
from . import normalization as norm

__all__ = ['Fields']


@dataclasses.dataclass
class Fields:
    x: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.rad)
    y: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.rad)
    weight: u.Quantity = dataclasses.field(default_factory=lambda: [1.0] * u.dimensionless_unscaled)
    decenter_x: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.dimensionless_unscaled)
    decenter_y: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.dimensionless_unscaled)
    vcx: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.dimensionless_unscaled)
    vcy: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.dimensionless_unscaled)
    van: u.Quantity = dataclasses.field(default_factory=lambda: [0.0] * u.rad)
    normalization: norm.Normalization = dataclasses.field(default_factory=lambda: norm.Rectangular())

    @classmethod
    def from_physical_pupil_size(
            cls,
            field_x: u.Quantity,
            field_y: u.Quantity,
            pupil_decenter_x: u.Quantity,
            pupil_decenter_y: u.Quantity,
            pupil_semi_axis_x: u.Quantity,
            pupil_semi_axis_y: u.Quantity,
            entrance_pupil_radius: u.Quantity,
    ) -> 'Fields':
        vdx, vdy = pupil.normalize_coordinates(pupil_decenter_x, pupil_decenter_y, entrance_pupil_radius)

        vcx = 1 - (pupil_semi_axis_x / entrance_pupil_radius)
        vcy = 1 - (pupil_semi_axis_y / entrance_pupil_radius)

        weight = [1.0] * u.dimensionless_unscaled
        van = [0] * u.deg

        return cls(field_x, field_y, weight, vdx, vdy, vcx, vcy, van)

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
        return np.broadcast(
            self.x,
            self.y,
            self.weight,
            self.vdx,
            self.vdy,
            self.vcx,
            self.vcy,
            self.van,
        ).shape[~0]

import dataclasses
import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u

from kgpy import math, optics

from . import Surface, Material, Aperture

__all__ = ['Standard']


@dataclasses.dataclass
class Standard(Surface):

    radius: u.Quantity = 0 * u.mm
    conic: u.Quantity = 0 * u.dimensionless_unscaled
    material: tp.Optional[Material] = None
    aperture: tp.Optional[Aperture] = None

    decenter_before: u.Quantity = [0, 0, 0] * u.m
    decenter_after: u.Quantity = [0, 0, 0] * u.m

    tilt_before: u.Quantity = [0, 0, 0] * u.deg
    tilt_after: u.Quantity = [0, 0, 0] * u.deg
    
    tilt_first: tp.Union[bool, npt.Array[bool]] = False

    @property
    def broadcasted_attrs(self):
        a = np.broadcast(
            super().broadcasted_attrs,
            self.radius,
            self.conic,
            self.decenter_before[..., 0],
            self.decenter_after[..., 0],
            self.tilt_before[..., 0],
            self.tilt_after[..., 0],
            self.tilt_first,
        )

        if self.aperture is not None:
            a = np.broadcast(a, self.aperture.broadcasted_attrs)

        return a


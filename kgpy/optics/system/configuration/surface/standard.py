import dataclasses
import numpy as np
import astropy.units as u

from kgpy import math, optics

from . import Surface, Material, Aperture

__all__ = ['Standard']


@dataclasses.dataclass
class Standard(Surface):

    radius: u.Quantity = 0 * u.mm
    conic: u.Quantity = 0 * u.dimensionless_unscaled
    material: Material = None
    aperture: Aperture = None
    pre_tilt_decenter: math.geometry.CoordinateSystem = None,
    post_tilt_decenter: math.geometry.CoordinateSystem = None,

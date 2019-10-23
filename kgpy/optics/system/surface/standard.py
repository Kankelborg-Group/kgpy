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
    material: tp.Union[tp.Optional[Material], npt.Array[tp.Optional[Material]]] = None
    aperture: tp.Union[tp.Optional[Aperture], npt.Array[tp.Optional[Aperture]]] = None

    decenter_before: u.Quantity = [0, 0, 0] * u.m
    decenter_after: u.Quantity = [0, 0, 0] * u.m

    tilt_before: u.Quantity = [0, 0, 0] * u.deg
    tilt_after: u.Quantity = [0, 0, 0] * u.deg

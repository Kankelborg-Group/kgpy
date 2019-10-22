import dataclasses
import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u

from kgpy import math, optics

from . import SurfaceArray, Material, Aperture

__all__ = ['Standard']


@dataclasses.dataclass
class Standard(SurfaceArray):

    radius: u.Quantity = [[0]] * u.mm
    conic: u.Quantity = [[0]] * u.dimensionless_unscaled
    material: npt.Array[tp.Optional[Material]] = np.array([[None]])
    aperture: npt.Array[tp.Optional[Aperture]] = np.array([[None]])

    decenter_before: u.Quantity = [[[0, 0, 0]]] * u.m
    decenter_after: u.Quantity = [[[0, 0, 0]]] * u.m

    tilt_before: u.Quantity = [[[0, 0, 0]]] * u.deg
    tilt_after: u.Quantity = [[[0, 0, 0]]] * u.deg

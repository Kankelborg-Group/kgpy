import dataclasses
import astropy.units as u

from kgpy import math

from . import SurfaceArray

__all__ = ['CoordinateBreak']


class CoordinateBreak:

    decenter: u.Quantity = [[[0, 0, 0]]] * u.m
    tilt: u.Quantity = [[[0, 0, 0]]] * u.deg

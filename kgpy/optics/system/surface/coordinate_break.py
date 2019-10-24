import dataclasses
import astropy.units as u

from kgpy import math

from . import Surface

__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class CoordinateBreak(Surface):

    decenter: u.Quantity = [0, 0, 0] * u.m
    tilt: u.Quantity = [0, 0, 0] * u.deg

    translation_first: bool = True

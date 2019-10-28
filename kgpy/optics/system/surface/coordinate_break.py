import dataclasses
import typing as tp
import nptyping as npt
import astropy.units as u

from . import Surface

__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class CoordinateBreak(Surface):

    decenter: u.Quantity = [0, 0, 0] * u.m
    tilt: u.Quantity = [0, 0, 0] * u.deg

    tilt_first: tp.Union[bool, npt.Array[bool]] = False

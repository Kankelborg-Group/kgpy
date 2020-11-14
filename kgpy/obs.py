import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import astropy.time
from kgpy import AutoAxis

__all__ = ['Obs']


class Axis(AutoAxis):
    def __init__(self):
        super().__init__()
        self.y = self.auto_axis_index()
        self.x = self.auto_axis_index()
        self.w = self.auto_axis_index()
        self.t = self.auto_axis_index(from_right=False)


@dataclasses.dataclass
class Obs:
    axis: typ.ClassVar[Axis] = Axis()
    intensity: u.Quantity
    time: astropy.time.Time
    exposure_length: u.Quantity
    wavelength: u.Quantity
    wcs: np.ndarray

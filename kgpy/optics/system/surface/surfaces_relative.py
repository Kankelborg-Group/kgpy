import dataclasses
import typing as tp
import numpy as np
import astropy.units as u

from . import surface, coordinate_break
from .. import mixin

__all__ = ['SurfacesRelative']


@dataclasses.dataclass
class SurfacesRelative(coordinate_break.CoordinateBreakMixin, mixin.Named):
    """
    This object lets you place a list of surfaces relative to the position of the current surface, and then return to
    the position of the current surface after the list of surfaces.
    This is useful if an optical component is represented as more than one surface and each surface needs to move in
    tandem.
    """

    surfaces: tp.List[surface.Surface] = dataclasses.field(default_factory=lambda: [])
    
    @property
    def all_surfaces(self) -> tp.List[surface.Surface]:
        return self.cbreak_before.surfaces + self.surfaces + self.cbreak_after.surfaces
    
    @property
    def cbreak_before(self):
        n = self.name + '.cb_before'
        return coordinate_break.ArbitraryDecenterZ(n, self.tilt, self.decenter, self.tilt_first)

    @property
    def cbreak_after(self):
        n = self.name + '.cb_after'
        return coordinate_break.ArbitraryDecenterZ(n, -self.tilt, -self.decenter, not self.tilt_first)


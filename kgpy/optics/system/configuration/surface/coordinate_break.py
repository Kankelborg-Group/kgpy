import dataclasses
import astropy.units as u

from kgpy import math

from . import SurfaceArray

__all__ = ['CoordinateBreak']


class CoordinateBreak(SurfaceArray):

    def __init__(self, *args,
                 tilt_decenter: math.geometry.CoordinateSystem = None,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)

        if tilt_decenter is None:
            tilt_decenter = math.geometry.CoordinateSystem()

        self._tilt_decenter = tilt_decenter

    @property
    def tilt_decenter(self) -> math.geometry.CoordinateSystem:
        return self._tilt_decenter

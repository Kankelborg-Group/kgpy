import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

import kgpy.typing.numpy as npt

from .. import mixin
from . import Surface


__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class CoordinateBreakMixin(mixin.ConfigBroadcast):
    """
    `CoordinateBreak` parameters split into a separate class so `ArbitraryDecenterZ` can inherit these parameters
    without all the other parameters of `Surface`.
    """

    tilt: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.deg)
    decenter: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.m)
    tilt_first: typ.Union[bool, npt.Array[bool]] = False

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.decenter[..., 0],
            self.tilt[..., 0],
            self.tilt_first,
        )


@dataclasses.dataclass
class CoordinateBreak(CoordinateBreakMixin, Surface):
    """
    Representation of a Zemax Coordinate Break.
    """

    @property
    def config_broadcast(self):

        return super().config_broadcast


@dataclasses.dataclass
class ArbitraryDecenterZ(CoordinateBreakMixin, mixin.Named):
    """
    Zemax doesn't allow decenters in the z-direction, instead they intend this concept to be represented by the
    `thickness` parameter.
    The problem with their representation is that the `thickness` parameter is always applied last and does not respect
    the `order` parameter.
    If you're trying to invert a 3D translation/rotation this is a problem since sometimes you need the translation
    in z applied first.
    The obvious way of fixing the problem is to define another surface before the coordinate break surface that can
    apply the z-translation first if needed.
    This is a class that acts as a normal `CoordinateBreak`, but is a composition of two coordinate breaks.
    It respects the `order` parameter for 3D translations.
    """

    @property
    def cbreak(self) -> CoordinateBreak:
        """
        The main surface corresponding to the coordinate break.
        """
        
        if self.tilt_first:
            return CoordinateBreak(self.tilt, self.decenter, self.tilt_first)
        
        else:
            t = self.tilt.copy()
            t[..., ~0] = 0
            return CoordinateBreak(t, self.decenter, self.tilt_first)

    @property
    def cbreak_z(self) -> CoordinateBreak:
        """
        The extra surface corresponding to the possible z-translation.
        """

        if self.tilt_first:
            return CoordinateBreak(self.tilt, self.decenter, self.tilt_first)

        else:
            t = self.tilt.copy()
            t[..., :~0] = 0
            return CoordinateBreak(t)
    
    @property
    def surfaces(self) -> typ. List[Surface]:
        """
        A list representation of the two surfaces that make up this arbitrary z decenter coordinate break.
        """
        return [
            self.cbreak_z,
            self.cbreak,
        ]
    
    # @property
    # def config_broadcast(self):
    #     return np.broadcast(
    #         super().config_broadcast,
    #         self.name,
    #     )
        
    

            


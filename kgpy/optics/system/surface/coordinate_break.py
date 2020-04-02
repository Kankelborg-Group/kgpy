import dataclasses
import numpy as np
from . import coordinate, surface

__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class Base:

    _tilt_decenter: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter(),
                                                                init=False, repr=False)


@dataclasses.dataclass
class CoordinateBreak(coordinate.TiltDecenter, Base, surface.Surface):
    """
    Representation of a Zemax Coordinate Break.
    """

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
        )

    @property
    def tilt_decenter(self):
        return self._tilt_decenter

    @property
    def tilt(self) -> coordinate.Tilt:
        return self.tilt_decenter.tilt

    @tilt.setter
    def tilt(self, value: coordinate.Tilt):
        self.tilt_decenter.tilt = value

    @property
    def decenter(self) -> coordinate.Decenter:
        return self.tilt_decenter.decenter

    @decenter.setter
    def decenter(self, value: coordinate.Decenter):
        self.tilt_decenter.decenter = value

    @property
    def tilt_first(self) -> coordinate.TiltFirst:
        return self.tilt_decenter.tilt_first

    @tilt_first.setter
    def tilt_first(self, value: coordinate.TiltFirst):
        self.tilt_decenter.tilt_first = value

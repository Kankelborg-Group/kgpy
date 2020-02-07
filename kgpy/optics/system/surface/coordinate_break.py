import dataclasses
import numpy as np

from .. import coordinate
from . import surface


__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class CoordinateBreak(surface.Surface):
    """
    Representation of a Zemax Coordinate Break.
    """
    _transform: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter())

    @property
    def tilt(self) -> coordinate.Tilt:
        return self._transform.tilt

    @tilt.setter
    def tilt(self, value: coordinate.Tilt):
        self._transform.tilt = value

    @property
    def decenter(self) -> coordinate.Decenter:
        return self._transform.decenter

    @decenter.setter
    def decenter(self, value: coordinate.Decenter):
        self._transform.decenter = value

    @property
    def tilt_first(self) -> bool:
        return self._transform.tilt_first

    @tilt_first.setter
    def tilt_first(self, value: bool):
        self._transform.tilt_first = value

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self._transform.config_broadcast
        )



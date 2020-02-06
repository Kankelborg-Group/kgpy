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
    transform: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter())

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.transform.config_broadcast
        )



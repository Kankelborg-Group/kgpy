import dataclasses
import numpy as np
import astropy.units as u
from .. import mixin

__all__ = ['Surface']


@dataclasses.dataclass
class Surface(mixin.Broadcastable, mixin.Named):
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    thickness: u.Quantity = 0 * u.mm
    is_stop: bool = False
    is_active: 'np.ndarray[bool]' = np.array(True)
    is_visible: 'np.ndarray[bool]' = np.array(True)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.thickness,
            self.is_active,
        )

    @property
    def thickness_vector(self):
        a = np.zeros(self.thickness.shape + (3,)) << self.thickness.unit
        a[..., ~0] = self.thickness
        return a

    def __iter__(self):
        yield self

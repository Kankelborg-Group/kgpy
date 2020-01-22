import dataclasses
import typing as typ
import numpy as np
import nptyping as npt
import astropy.units as u

from .. import mixin

__all__ = ['Surface']


@dataclasses.dataclass
class Surface(mixin.Named):
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """
    thickness: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.mm)
    is_active: typ.Union[bool, npt.Array[bool]] = True

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.thickness,
            self.is_active,
        )

    @property
    def surfaces(self) -> typ.List['Surface']:
        return [self]


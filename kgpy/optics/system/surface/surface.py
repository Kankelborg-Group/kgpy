import dataclasses
import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u

__all__ = ['Surface']


@dataclasses.dataclass
class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    name: tp.Union[str, npt.Array[str]] = ''
    thickness: u.Quantity = 0 * u.mm
    is_active: tp.Union[bool, npt.Array[bool]] = True

    @property
    def config_broadcast(self):
        return np.broadcast(
            self.name,
            self.thickness,
            self.is_active,
        )


import dataclasses
import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u

from . import Material, Aperture

__all__ = ['Surface']


@dataclasses.dataclass
class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    name: tp.Union[str, npt.Array[str]] = ''
    is_stop: tp.Union[bool, npt.Array[bool]] = False
    thickness: u.Quantity = 0 * u.mm



import dataclasses
import typing as tp
import numpy as np
import astropy.units as u
from beautifultable import BeautifulTable

from kgpy import optics, math

from . import Aperture, Material

__all__ = ['Surface']


@dataclasses.dataclass
class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    name: str = ''
    is_stop: bool = False
    thickness: u.Quantity = None

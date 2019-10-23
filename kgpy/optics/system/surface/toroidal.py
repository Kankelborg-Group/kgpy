import dataclasses
import astropy.units as u

from . import Standard

__all__ = ['Toroidal']


@dataclasses.dataclass
class Toroidal(Standard):

    radius_of_rotation = 0 * u.mm

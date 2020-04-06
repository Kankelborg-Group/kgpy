import dataclasses
import astropy.units as u

from .. import mixin

__all__ = ['Material', 'NoMaterial', 'Mirror']


@dataclasses.dataclass
class Material(mixin.Broadcastable):
    pass


@dataclasses.dataclass
class NoMaterial(Material):
    pass


@dataclasses.dataclass
class Mirror(Material):

    thickness: u.Quantity = 0 * u.mm

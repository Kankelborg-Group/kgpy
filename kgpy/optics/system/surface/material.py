import abc
import dataclasses
import astropy.units as u

from .. import mixin

__all__ = ['Material', 'NoMaterial', 'Mirror']


class Material(abc.ABC, mixin.ConfigBroadcast):

    @abc.abstractmethod
    def __str__(self):
        ...


class NoMaterial(Material):

    def __str__(self):
        return ''


@dataclasses.dataclass
class Mirror(Material):
    
    thickness: u.Quantity = 0 * u.mm
    
    def __str__(self):
        return 'MIRROR'

import abc
import dataclasses
import astropy.units as u

from .. import mixin

__all__ = ['Material', 'NoMaterial', 'Mirror']


class Material(abc.ABC, mixin.ConfigBroadcast):

    @property
    @abc.abstractmethod
    def to_str(self):
        ...


class NoMaterial(Material):

    @property
    def to_str(self):
        return ''


@dataclasses.dataclass
class Mirror(Material):
    
    thickness: u.Quantity
    
    @property
    def to_str(self):
        return 'MIRROR'

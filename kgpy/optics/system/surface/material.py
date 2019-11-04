import abc
import dataclasses
import astropy.units as u

__all__ = ['Material', 'Mirror']


@dataclasses.dataclass
class Material:
    
    @property
    @abc.abstractmethod
    def name(self):
        return ''


@dataclasses.dataclass
class Mirror(Material):
    
    thickness: u.Quantity
    
    @property
    def name(self):
        return 'MIRROR'


from typing import List
import astropy.units as u

from kgpy import optics

from .item import Item

__all__ = ['Array']


class Array:
    
    def __init__(self):
        
        self.items = []       # type: 
        
    def append(self, wavl: Item):
        
        self.items.append(wavl)
        
    def promote_to_zmx(self, zos_arr: 'optics.zemax.ZOSAPI.SystemData.IWavelengths') -> 'optics.zemax.wavelength.Array':

        a = optics.zemax.system.wavelength.Array(zos_arr)
        
        for item in self.items:
            a.append(item)
            
        return a


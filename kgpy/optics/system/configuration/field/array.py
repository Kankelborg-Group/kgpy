
from typing import List

from kgpy import optics

from .item import Item

__all__ = ['Array']


class Array:
    
    def __init__(self):
        
        self.items = []         # type: List[Item]
        
    def append(self, field: Item):
        
        self.items.append(field)

    def promote_to_zmx(self, zos_sys: 'optics.zemax.ZOSAPI.IOpticalSystem') -> 'optics.zemax.field.Array':
        a = kgpy.optics.zemax.system.configuration.field.Array(zos_sys)

        for item in self.items:
            a.append(item)

        return a

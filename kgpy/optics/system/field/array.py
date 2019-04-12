
from typing import List

from kgpy import optics

from .item import Item

__all__ = ['Array']


class Array:
    
    def __init__(self):
        
        self.items = []         # type: List[Item]
        
    def append(self, field: Item):
        
        self.items.append(field)

    def promote_to_zmx(self, zos_arr: 'optics.zemax.ZOSAPI.SystemData.IFields') -> 'optics.zemax.field.Array':
        a = optics.zemax.system.field.Array(zos_arr)

        for item in self.items:
            a.append(item)

        return a

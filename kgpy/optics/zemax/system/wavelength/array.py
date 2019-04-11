
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.system.wavelength.array import Array as ArrayBase, Item as ItemBase

from .item import Item

__all__ = ['Array']


class Array(ArrayBase):

    def __init__(self, zos_arr: ZOSAPI.SystemData.IWavelengths):

        self.zos_arr = zos_arr

        ArrayBase.__init__(self)

    def append(self, wavl: ItemBase):

        if len(self.items) < 1:
            zos_wavl = self.zos_arr.GetWavelength(1)
        
        else:
            zos_wavl = self.zos_arr.AddWavelength(0.5, 1.0)

        w = Item(wavl.wavelength, zos_wavl)
        w.weight = wavl.weight

        ArrayBase.append(self, w)


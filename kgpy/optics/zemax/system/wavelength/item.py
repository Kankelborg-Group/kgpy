
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.system.wavelength.item import Item as ItemBase

__all__ = ['Item']


class Item(ItemBase):
    
    units = u.um

    def __init__(self, wavelength: u.Quantity, zos_wavl: ZOSAPI.SystemData.IWavelength):

        self.zos_wavl = zos_wavl

        ItemBase.__init__(self, wavelength)

    @property
    def wavelength(self) -> u.Quantity:
        return self.zos_wavl.Wavelength * self.units

    @wavelength.setter
    def wavelength(self, value: u.Quantity):
        self.zos_wavl.Wavelength = value.to(self.units).value
        
    @property
    def weight(self) -> float:
        return self.zos_wavl.Weight
    
    @weight.setter
    def weight(self, value: float):
        self.zos_wavl.Weight = value


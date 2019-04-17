
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.system.wavelength.item import Item as ItemBase

__all__ = ['Item']


class Item(ItemBase):
    
    units = u.um

    def __init__(self, wavelength: u.Quantity, zos_wavl: ZOSAPI.SystemData.IWavelength, zos_sys: ZOSAPI.IOpticalSystem):

        self.zos_wavl = zos_wavl
        self.zos_mce = zos_sys.MCE

        self._prep_mce()

        ItemBase.__init__(self, wavelength)
        
    def _prep_mce(self):
        
        self._wavlength_op = self.zos_mce.AddOperand()
        self._wavlength_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.WAVE)
        self._wavlength_op.Param1 = self.zos_wavl.WavelengthNumber

        self._weight_op = self.zos_mce.AddOperand()
        self._weight_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.WLWT)
        self._weight_op.Param1 = self.zos_wavl.WavelengthNumber

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


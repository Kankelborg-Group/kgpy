from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.system.configuration.wavelength import WavelengthList as ArrayBase, Wavelength as ItemBase

from .item import Wavelength

__all__ = ['WavelengthList']


class WavelengthList(ArrayBase):

    def __init__(self, zos_sys: ZOSAPI.IOpticalSystem):

        self.zos_sys = zos_sys

        self.zos_arr = self.zos_sys.SystemData.Wavelengths

        ArrayBase.__init__(self)

    def append(self, wavl: ItemBase):

        if len(self.items) < 1:
            zos_wavl = self.zos_arr.GetWavelength(1)
        
        else:
            zos_wavl = self.zos_arr.AddWavelength(0.5, 1.0)

        w = Wavelength(wavl.wavelength, zos_wavl, self.zos_sys)
        w.weight = wavl.weight

        ArrayBase.append(self, w)



from kgpy.optics.zemax import ZOSAPI

__all__ = ['IWavelengths']


class IWavelengths:

    NumberOfWavelengths = None          # type: int
    
    def AddWavelength(self, Wavelength: float, Weight: float) -> 'ZOSAPI.SystemData.IWavelength':
        pass

    def GaussianQuadrature(self, minWave: float, maxWave: float, numSteps: 'ZOSAPI.SystemData.QuadratureSteps'):
        pass

    def GetWavelength(self, position: int) -> 'ZOSAPI.SystemData.IWavelength':
        pass

    def RemoveWavelength(self, position: int) -> bool:
        pass

    def SelectWavelengthPreset(self, preset: 'ZOSAPI.SystemData.WavelengthPreset') -> bool:
        pass

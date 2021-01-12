
from win32com.client import constants

__all__ = ['RaysType']


class RaysTypeBase:
    
    @property
    def Real(self) -> 'RaysType':
        return constants.RaysType_Real
    
    @property
    def Paraxial(self) -> 'RaysType':
        return constants.RaysType_Paraxial


RaysType = RaysTypeBase()

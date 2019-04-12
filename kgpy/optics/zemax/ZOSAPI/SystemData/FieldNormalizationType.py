
from win32com.client import constants

__all__ = ['FieldNormalizationType']


class FieldNormalizationTypeBase:
    
    @property
    def Radial(self):
        return constants.FieldNormalizationType_Radial
    
    @property
    def Rectangular(self):
        return constants.FieldNormalizationType_Rectangular
    

FieldNormalizationType = FieldNormalizationTypeBase()

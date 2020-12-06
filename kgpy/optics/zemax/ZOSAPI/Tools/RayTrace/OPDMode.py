
from win32com.client import constants

__all__ = ['OPDMode']


class OPDModeBase:
    
    @property
    def None_(self):
        return constants.OPDMode_None
    
    
OPDMode = OPDModeBase()

from win32com.client import constants

__all__ = ['SubstrateType']


class SubstrateTypeBase:
    
    @property
    def none(self):
        return constants.SubstrateType_None
    
    @property
    def Flat(self):
        return constants.SubstrateType_Flat
    
    @property
    def Curved(self):
        return constants.SubstrateType_Curved
    
    
SubstrateType = SubstrateTypeBase()
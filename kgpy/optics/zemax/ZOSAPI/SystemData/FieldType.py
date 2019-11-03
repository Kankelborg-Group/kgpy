
from win32com.client import constants

__all__ = ['']

class FieldTypeBase:
    
    @property
    def Angle(self) -> 'FieldType':
        return constants.FieldType_Angle
    
    @property
    def ObjectHeight(self) -> 'FieldType':
        return constants.FieldType_ObjectHeight
    
    @property
    def ParaxialImageHeight(self) -> 'FieldType':
        return constants.FieldType_ParaxialImageHeight
    
    @property
    def RealImageHeight(self) -> 'FieldType':
        return constants.FieldType_RealImageHeight
    
    @property
    def TheodoliteAngle(self) -> 'FieldType':
        return constants.FieldType_TheodoliteAngle


FieldType = FieldTypeBase()


from kgpy.optics.zemax import ZOSAPI

__all__ = ['IFields']


class IFields:

    Normalization = None        # type: ZOSAPI.SystemData.FieldNormalizationType
    NumberOfFields = None       # int

    def AddField(self, X: float, Y: float, Weight: float) -> 'ZOSAPI.SystemData.IField':
        pass

    def ClearVignetting(self):
        pass

    def GetField(self, position: int) -> 'ZOSAPI.SystemData.IField':
        pass

    def GetFieldType(self) -> 'ZOSAPI.SystemData.FieldType':
        pass

    def MakeEqualAreaFields(self, numberOfFields: int, maximumField: float) -> bool:
        pass

    def RemoveField(self, position: int) -> bool:
        pass

    def SetFieldType(self, type: 'ZOSAPI.SystemData.FieldType') -> bool:
        pass

    def SetVignetting(self):
        pass

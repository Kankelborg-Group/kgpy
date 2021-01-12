
from kgpy.optics.zemax import ZOSAPI

__all__ = ['IAS_']


class IAS_:

    def Load(self) -> None:
        pass

    def LoadFrom(self, settingsFile: str) -> bool:
        pass

    def ModifySettings(self, settingsFile: str, typeCode: str, newValue: str) -> bool:
        pass

    def Reset(self) -> None:
        pass

    def Save(self) -> None:
        pass

    def SaveTo(self, settingsFile: str) -> bool:
        pass

    def Verify(self) -> 'ZOSAPI.Analysis.IMessages':
        pass

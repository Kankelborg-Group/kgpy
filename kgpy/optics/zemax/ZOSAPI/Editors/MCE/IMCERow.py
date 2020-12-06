
from typing import List

from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.ZOSAPI.Editors.IEditorRow import IEditorRow
__all__ = ['IMCERow']


class IMCERow(IEditorRow):
    
    IsActive = None             # type: bool
    OperandNumber = None        # type: int
    Param1 = None               # type: int
    Param1Enabled = None        # type: bool
    Param2 = None               # type: int
    Param2Enabled = None        # type: bool
    Param3 = None               # type: int
    Param3Enabled = None        # type: bool
    RowColor = None             # type: ZOSAPI.Common.ZemaxColor
    Type = None                 # type: ZOSAPI.Editors.MCE.MultiConfigOperandType
    TypeName = None             # type: str

    def AvailableConfigOperandTypes(self) -> List['ZOSAPI.Editors.MCE.MultiConfigOperandType']:
        pass

    def ChangeType(self, type_: 'ZOSAPI.Editors.MCE.MultiConfigOperandType') -> bool:
        pass

    def GetOperandCell(self, configuration: int) -> 'ZOSAPI.Editors.IEditorCell':
        pass

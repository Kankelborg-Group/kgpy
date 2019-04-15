
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.ZOSAPI.Editors.IEditor import IEditor

__all__ = ['IMultiConfigEditor']


class IMultiConfigEditor(IEditor):

    CurrentConfiguration = None         # type: int
    FirstConfiguration = None           # type: int
    LastConfiguration = None            # type: int
    NumberOfConfigurations = None       # type: int
    NumberOfOperands = None             # type: int
    RowToOperandOffset = None           # type: int

    def AddConfiguration(self, withPickups: bool) -> bool:
        pass

    def AddOperand(self) -> 'ZOSAPI.Editors.MCE.IMCERow':
        pass

    def CopyOperands(self, fromOperandNumber: int, NumberOfOperands: int, toOperandNumber: int) -> int:
        pass

    def CopyOperandsFrom(self, fromEditor: 'IMultiConfigEditor', fromOperandNumber: int, NumberOfOperands: int,
                         toOperandNumber: int) -> int:
        pass

    def DeleteAllConfigurations(self) -> bool:
        pass

    def DeleteConfiguration(self, configurationNumber: int) -> bool:
        pass

    def GetOperandAt(self, OperandNumber: int) -> 'ZOSAPI.Editors.MCE.IMCERow':
        pass

    def HideMCE(self):
        pass

    def InsertConfiguration(self, configurationNumber: int, wihtPickups: bool) -> bool:
        pass

    def InsertNewOperandAt(self, OperandNumber: int) -> 'ZOSAPI.Editors.MCE.IMCERow':
        pass

    def MakeSingleConfiguration(self):
        pass

    def MakeSingleConfigurationOpt(self, deleteMFEOperands: bool):
        pass

    def NextConfiguration(self) -> bool:
        pass

    def PrevConfiguration(self) -> bool:
        pass

    def RemoveOperandAt(self, OperandNumber: int) -> bool:
        pass

    def RemoveOperandsAt(self, OperandNumber: int, numOperands: int) -> int:
        pass

    def SetCurrentConfiguration(self, configurationNumber: int) -> bool:
        pass

    def ShowMCE(self) -> bool:
        pass

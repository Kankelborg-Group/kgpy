import typing as tp
from kgpy.optics.zemax import ZOSAPI

__all__ = ['IEditorCell']


class IEditorCell:

    Col: int
    DataType: 'ZOSAPI.Editors.CellDataType'
    DoubleValue: float
    Header: str
    IntegerValue: int
    IsActive: bool
    IsReadOnly: bool
    Row: 'ZOSAPI.Editors.IEditorRow'
    Solve: 'ZOSAPI.Editors.SolveType'
    Value: str

    def CreateSolveType(self, type_: 'ZOSAPI.Editors.SolveType') -> 'ZOSAPI.Editors.ISolveData':
        pass

    def FillAvailableSolveTypes(self, Length: int, solves: tp.List['ZOSAPI.Editors.SolveType']):
        pass

    def GetAvailableSolveTypes(self) -> tp.List['ZOSAPI.Editors.SolveType']:
        pass

    def GetNumberOfSolveTypes(self) -> int:
        pass

    def GetSolveData(self) -> 'ZOSAPI.Editors.ISolveData':
        pass

    def IsSolveTypeSupported(self, st: 'ZOSAPI.Editors.SolveType') -> bool:
        pass

    def MakeSolveFixed(self) -> bool:
        pass

    def MakeSolveVariable(self) -> bool:
        pass

    def SetSolveData(self, Data: 'ZOSAPI.Editors.ISolveData') -> 'ZOSAPI.Editors.SolveStatus':
        pass

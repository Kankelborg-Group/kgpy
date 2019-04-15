
from kgpy.optics.zemax import ZOSAPI

__all__ = ['IEditor']


class IEditor:

    Editor = None           # type: ZOSAPI.Editors.EditorType
    MaxColumn = None        # type: int
    MinColumn = None        # type: int
    NumberOfRows = None     # type: int

    def AddRow(self) -> 'ZOSAPI.Editors.IEditorRow':
        pass

    def DeleteAllRows(self) -> int:
        pass

    def DeleteRowAt(self, pos: int) -> bool:
        pass

    def DeleteRowsAt(self, pos: int, numberOfRows: int) -> int:
        pass

    def GetRowAt(self, pos: int) -> 'ZOSAPI.Editors.IEditorRow':
        pass

    def HideEditor(self):
        pass

    def InsertRowAt(self, pos: int) -> 'ZOSAPI.Editors.IEditorRow':
        pass

    def ShowEditor(self) -> bool:
        pass

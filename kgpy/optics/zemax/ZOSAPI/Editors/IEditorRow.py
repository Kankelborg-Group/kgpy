
from kgpy.optics.zemax import ZOSAPI

__all__ = 'IEditorRow'


class IEditorRow:

    Editor = None           # type: ZOSAPI.Editors.IEditor
    IsValidRow = None       # type: bool
    RowIndex = None         # type: int
    RowTypeName = None      # type: str

    def GetCellAt(self, pos: int) -> 'ZOSAPI.Editors.IEditorCell':
        pass

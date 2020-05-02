import dataclasses
import typing as typ
import win32com.client
from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import surface
from . import coordinate

__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class CoordinateBreak(system.surface.CoordinateBreak, surface.Surface):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.transform = self.transform

    @property
    def _lde_row_type(self) -> ZOSAPI.Editors.LDE.SurfaceType:
        return ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak

    @property
    def _lde_row_data(self) -> ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak:
        return win32com.client.CastTo(self._lde_row.SurfaceData, ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak.__name__)

    @property
    def _lde_row(self) -> ZOSAPI.Editors.LDE.ILDERow[ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak]:
        return super()._lde_row

    @property
    def transform(self) -> coordinate.TiltDecenter:
        return self._transform

    @transform.setter
    def transform(self, value: coordinate.TiltDecenter):
        if not isinstance(value, coordinate.TiltDecenter):
            value = coordinate.TiltDecenter.promote(value)
        value._composite = self
        self._transform = value

import dataclasses
import typing as typ
from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import surface
from . import coordinate

__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class CoordinateBreak(system.surface.CoordinateBreak, surface.Surface):

    def _get_type(self) -> ZOSAPI.Editors.LDE.SurfaceType:
        return ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak

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

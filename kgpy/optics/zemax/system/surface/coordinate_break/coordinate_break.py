import dataclasses
from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import surface
from . import coordinate

__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class CoordinateBreak(system.surface.CoordinateBreak, surface.Surface):

    tilt_decenter: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter(),
                                                               init=False, repr=False)

    @property
    def lde_row(self) -> ZOSAPI.Editors.LDE.ILDERow[ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak]:
        return super().lde_row

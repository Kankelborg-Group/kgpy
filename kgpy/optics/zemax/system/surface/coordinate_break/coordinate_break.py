import dataclasses
from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import surface
from . import coordinate

__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class Base:

    tilt_decenter: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter(),
                                                               init=False, repr=False)

    tilt: coordinate.Tilt = dataclasses.field(default_factory=lambda: coordinate.Tilt())
    decenter: coordinate.Decenter = dataclasses.field(default_factory=lambda: coordinate.Decenter())


@dataclasses.dataclass
class CoordinateBreak(system.surface.coordinate_break.Base, surface.Surface):

    @property
    def lde_row(self) -> ZOSAPI.Editors.LDE.ILDERow[ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak]:
        return super().lde_row

    @property
    def tilt(self) -> coordinate.Tilt:
        return self.tilt_decenter.tilt

    @tilt.setter
    def tilt(self, value: coordinate.Tilt):
        self.tilt_decenter.tilt = value

    @property
    def decenter(self) -> coordinate.Decenter:
        return self.tilt_decenter.decenter

    @decenter.setter
    def decenter(self, value: coordinate.Decenter):
        self.tilt_decenter.decenter = value

    @property
    def tilt_first(self) -> bool:
        return self.tilt_decenter.tilt_first

    @tilt_first.setter
    def tilt_first(self, value: bool):
        self.tilt_decenter.tilt_first = value

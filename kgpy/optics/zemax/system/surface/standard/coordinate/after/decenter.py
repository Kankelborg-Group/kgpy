import dataclasses

from kgpy.optics.zemax import ZOSAPI
from ..... import configuration, surface
from . import tilt_decenter

__all__ = ['Decenter']


@dataclasses.dataclass
class Decenter(surface.coordinate.Decenter[tilt_decenter.TiltDecenter[surface.Surface]]):

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CADX,
        ),
        init=False,
        repr=False,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CADY,
        ),
        init=False,
        repr=False,
    )

    def _x_setter(self, value: float):
        self.parent.parent.lde_row.TiltDecenterData.AfterSurfaceDecenterX = value

    def _y_setter(self, value: float):
        self.parent.parent.lde_row.TiltDecenterData.AfterSurfaceDecenterY = value

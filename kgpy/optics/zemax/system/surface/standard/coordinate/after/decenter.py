import dataclasses

from kgpy.optics.zemax import ZOSAPI
from ..... import configuration
from .... import coordinate
from .. import after

__all__ = ['Decenter']


@dataclasses.dataclass
class Decenter(coordinate.Decenter['after.tilt_decenter.TiltDecenter[surface.Surface]']):

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.CADX,
        ),
        init=False,
        repr=False,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.CADY,
        ),
        init=False,
        repr=False,
    )

    def _x_setter(self, value: float):
        self.composite.composite.lde_row.TiltDecenterData.AfterSurfaceDecenterX = value

    def _y_setter(self, value: float):
        self.composite.composite.lde_row.TiltDecenterData.AfterSurfaceDecenterY = value

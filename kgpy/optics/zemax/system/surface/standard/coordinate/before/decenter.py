import dataclasses

from kgpy.optics.zemax import ZOSAPI
from ..... import configuration
from .... import coordinate
from .. import before

__all__ = ['Decenter']


@dataclasses.dataclass
class Decenter(coordinate.Decenter['before.tilt_decenter.TiltDecenter[surface.Standard]']):

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDX,
        ),
        init=False,
        repr=False,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDY,
        ),
        init=False,
        repr=False,
    )

    def _x_setter(self, value: float):
        self.parent.parent.lde_row.TiltDecenterData.BeforeSurfaceDecenterX = value

    def _y_setter(self, value: float):
        self.parent.parent.lde_row.TiltDecenterData.BeforeSurfaceDecenterY = value

import dataclasses

from kgpy.optics.zemax import ZOSAPI
from ..... import configuration, surface
from . import tilt_decenter

__all__ = ['TiltFirst']


@dataclasses.dataclass
class TiltFirst(surface.coordinate.TiltFirst[tilt_decenter.TiltDecenter[surface.Standard]]):

    _value_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CAOR,
        ),
        init=False,
        repr=False,
    )

    def _value_setter(self, value: int):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceOrder = value

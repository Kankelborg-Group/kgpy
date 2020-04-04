import dataclasses
from kgpy.optics.zemax import ZOSAPI
from ..... import configuration
from .... import coordinate
from ... import standard
from . import tilt as tilt_, decenter as decenter_

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class TiltDecenter(coordinate.TiltDecenter[standard.Standard], ):

    _tilt_first_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.CAOR,
        ),
        init=False,
        repr=False,
    )

    tilt: tilt_.Tilt = dataclasses.field(default_factory=lambda: tilt_.Tilt())
    decenter: decenter_.Decenter = dataclasses.field(default_factory=lambda: decenter_.Decenter())

    def _tilt_first_setter(self, value: int):
        self.composite.lde_row.TiltDecenterData.AfterSurfaceOrder = value

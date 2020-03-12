import dataclasses
import numpy as np

from kgpy.optics.system import mixin
from . import Tilt, InverseTilt, Translate, InverseTranslate, TiltFirst, InverseTiltFirst

__all__ = ['Transform', 'InverseTransform']


@dataclasses.dataclass
class Base(mixin.ConfigBroadcast):
    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    translate: Translate = dataclasses.field(default_factory=lambda: Translate())
    tilt_first: TiltFirst = dataclasses.field(default_factory=lambda: TiltFirst())


class Transform(Base):

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.tilt.config_broadcast,
            self.translate.config_broadcast,
        )

    def __invert__(self):
        return type(self)(
            self.tilt.__invert__(),
            self.translate.__invert__(),
            not self.tilt_first,
        )


@dataclasses.dataclass
class InverseTransform:

    _transform: Transform

    @property
    def config_broadcast(self):
        return self._transform.config_broadcast

    @property
    def tilt(self) -> InverseTilt:
        return InverseTilt(self._transform.tilt)
    
    @property
    def translate(self) -> InverseTranslate:
        return InverseTranslate(self._transform.translate)
    
    @property
    def tilt_first(self) -> InverseTiltFirst:
        return InverseTiltFirst(self._transform.tilt_first)

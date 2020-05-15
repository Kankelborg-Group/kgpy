import abc
import dataclasses
import numpy as np
import astropy.units as u
from ... import mixin

__all__ = ['Aperture']


@dataclasses.dataclass
class Aperture(mixin.ZemaxCompatible, mixin.Broadcastable, abc.ABC):

    def to_zemax(self) -> 'Aperture':
        from kgpy.optics import zemax
        return zemax.system.surface.aperture.Aperture()

    @abc.abstractmethod
    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        pass


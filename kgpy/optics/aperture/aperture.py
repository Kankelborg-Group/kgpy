import abc
import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import kgpy.mixin
from .. import ZemaxCompatible

__all__ = ['Aperture']


@dataclasses.dataclass
class Aperture(ZemaxCompatible, kgpy.mixin.Broadcastable, abc.ABC):

    num_samples: int = 100

    def to_zemax(self) -> 'Aperture':
        from kgpy.optics import zemax
        return zemax.system.surface.aperture.Aperture()

    @abc.abstractmethod
    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def edges(self) -> typ.Optional[u.Quantity]:
        pass

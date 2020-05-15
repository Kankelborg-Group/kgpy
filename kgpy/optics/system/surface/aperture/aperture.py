import abc
import dataclasses
import astropy.units as u
from ... import mixin

__all__ = ['Aperture']


@dataclasses.dataclass
class Aperture(mixin.ZemaxCompatible, mixin.Broadcastable, abc.ABC):

    def to_zemax(self) -> 'Aperture':
        from kgpy.optics import zemax
        return zemax.system.surface.aperture.Aperture()

    @abc.abstractmethod
    def check_points(self, points: u.Quantity):
        pass


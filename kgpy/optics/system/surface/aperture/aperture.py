import dataclasses

from ... import mixin

__all__ = ['Aperture']


@dataclasses.dataclass
class Aperture(mixin.ZemaxCompatible, mixin.Broadcastable):

    def to_zemax(self) -> 'Aperture':
        from kgpy.optics import zemax
        return zemax.system.surface.aperture.Aperture()

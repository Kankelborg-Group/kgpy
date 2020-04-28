import dataclasses
from ... import mixin

__all__ = ['Material', 'NoMaterial']


@dataclasses.dataclass
class Material(mixin.ZemaxCompatible, mixin.Broadcastable):
    pass


@dataclasses.dataclass
class NoMaterial(Material):
    def to_zemax(self) -> 'NoMaterial':
        from kgpy.optics import zemax
        return zemax.system.surface.material.NoMaterial()

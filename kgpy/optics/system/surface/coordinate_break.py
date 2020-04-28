import dataclasses
import typing as typ
from . import coordinate, surface

__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class CoordinateBreak(surface.Surface):

    transform: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter())

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'transform': self.transform
        })
        return args

    def to_zemax(self) -> 'CoordinateBreak':
        from kgpy.optics import zemax
        return zemax.system.surface.CoordinateBreak(**self.__init__args)

import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from . import Transform, TransformList, Decenter, TiltXYZ

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class TiltDecenter(Transform):
    tilt: typ.Optional[TiltXYZ] = None
    decenter: typ.Optional[Decenter] = None
    tilt_first: bool = False

    @classmethod
    def promote(cls, value: 'TiltDecenter'):
        return cls(value.tilt, value.decenter, value.tilt_first)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.tilt.config_broadcast,
            self.decenter.config_broadcast,
        )

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'tilt': self.tilt_first,
            'decenter': self.decenter,
            'tilt_first': self.tilt_first,
        })
        return args

    @property
    def _transform(self) -> 'TransformList':
        transform = TransformList([self.decenter, self.tilt])
        if self.tilt_first:
            transform.reverse()
        return transform

    def __invert__(self) -> 'TransformList':
        return self._transform.__invert__()

    def __call__(
            self,
            value: u.Quantity,
            use_rotations: bool = True,
            use_translations: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        return self._transform(value, use_rotations, use_translations, num_extra_dims)

    def copy(self) -> 'TiltDecenter':
        tilt = self.tilt
        if tilt is not None:
            tilt = tilt.copy()
        decenter = self.decenter
        if decenter is not None:
            decenter = decenter.copy()
        return TiltDecenter(
            tilt=tilt,
            decenter=decenter,
            tilt_first=self.tilt_first,
        )

import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from . import Transform, TransformList, TiltXYZ, Translate, TiltDecenter

__all__ = ['TiltTranslate']


@dataclasses.dataclass
class TiltTranslate(Transform):
    tilt: typ.Optional[TiltXYZ] = None
    translate: typ.Optional[Translate] = None
    tilt_first: bool = False

    @classmethod
    def from_tilt_decenter(
            cls,
            tilt_decenter: TiltDecenter = None,
            z=0 * u.mm,
    ):
        return cls(
            tilt=tilt_decenter.tilt.copy(),
            translate=Translate(
                x=tilt_decenter.decenter.x.copy(),
                y=tilt_decenter.decenter.y.copy(),
                z=z,
            )
        )

    @classmethod
    def promote(cls, value: 'TiltTranslate'):
        return cls(value.tilt, value.translate, value.tilt_first)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.tilt.config_broadcast,
            self.translate.config_broadcast,
        )

    @property
    def _transform(self) -> 'TransformList':
        transform = TransformList([self.translate, self.tilt])
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

    def copy(self) -> 'TiltTranslate':
        tilt = self.tilt
        if tilt is not None:
            tilt = tilt.copy()
        translate = self.translate
        if translate is not None:
            translate = translate.copy()
        return TiltTranslate(
            tilt=tilt,
            translate=translate,
            tilt_first=self.tilt_first,
        )


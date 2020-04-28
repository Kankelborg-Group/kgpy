import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from . import coordinate, Surface, material as material_, aperture as aperture_

__all__ = ['Standard']

MaterialT = typ.TypeVar('MaterialT', bound=material_.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture_.Aperture)


@dataclasses.dataclass
class Standard(
    typ.Generic[MaterialT, ApertureT],
    Surface,
):

    radius: u.Quantity = np.inf * u.mm
    conic: u.Quantity = 0 * u.dimensionless_unscaled
    material: MaterialT = dataclasses.field(default_factory=lambda: material_.NoMaterial())
    aperture: ApertureT = dataclasses.field(default_factory=lambda: aperture_.NoAperture())
    transform_before: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter())
    transform_after: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter())

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'radius': self.radius,
            'conic': self.conic,
            'material': self.material.to_zemax(),
            'aperture': self.aperture.to_zemax(),
            'transform_before': self.transform_before,
            'transform_after': self.transform_after,
        })
        return args

    def to_zemax(self) -> 'Standard':
        from kgpy.optics import zemax
        return zemax.system.surface.Standard(**self.__init__args)


    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius,
            self.conic,
            self.material.config_broadcast,
            self.aperture.config_broadcast,
            self.transform_before.config_broadcast,
            self.transform_after.config_broadcast,
        )

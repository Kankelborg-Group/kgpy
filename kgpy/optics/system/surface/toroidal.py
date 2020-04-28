import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from . import Standard, material, aperture

__all__ = ['Toroidal']

MaterialT = typ.TypeVar('MaterialT', bound=material.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture.Aperture)


@dataclasses.dataclass
class Toroidal(Standard[MaterialT, ApertureT]):

    radius_of_rotation: u.Quantity = 0 * u.mm

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'radius_of_rotation': self.radius_of_rotation,
        })
        return args

    def to_zemax(self) -> 'Toroidal':
        from kgpy.optics import zemax
        return zemax.system.surface.Toroidal(**self.__init__args)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius_of_rotation,
        )
import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import kgpy.vector
from kgpy.vector import x, y, z
from .. import Rays, coordinate
from . import surface

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

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.transform.config_broadcast,
        )

    def to_zemax(self) -> 'CoordinateBreak':
        from kgpy.optics import zemax
        return zemax.system.surface.CoordinateBreak(**self.__init__args)

    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return 0 * u.mm

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return u.Quantity([0, 0, 1])

    @property
    def _rays_output(self) -> typ.Optional[Rays]:
        if self.rays_input is None:
            return None
        rays = self.rays_input.copy()
        rays.position[z] -= self.previous_surface.thickness_eff
        return rays.tilt_decenter(~self.transform)

    @property
    def pre_transform(self) -> coordinate.Transform:
        return coordinate.Transform.from_tilt_decenter(self.transform)

    @property
    def post_transform(self) -> coordinate.Transform:
        return coordinate.Transform(translate=coordinate.Translate(z=self.thickness))

    def copy(self) -> 'CoordinateBreak':
        return CoordinateBreak(
            name=self.name.copy(),
            thickness=self.thickness.copy(),
            is_active=self.is_active.copy(),
            is_visible=self.is_visible.copy(),
            transform=self.transform.copy(),
        )

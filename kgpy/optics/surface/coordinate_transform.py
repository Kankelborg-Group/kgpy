import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.vector
from kgpy.vector import x, y, z
from .. import Rays, coordinate
from . import surface

__all__ = ['CoordinateTransform']


@dataclasses.dataclass
class CoordinateTransform(surface.Surface):

    transform: typ.Optional[coordinate.Transform] = None

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

    def to_zemax(self) -> 'CoordinateTransform':
        from kgpy.optics import zemax
        return zemax.system.surface.CoordinateBreak(**self.__init__args)

    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return 0 * u.mm

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return u.Quantity([0, 0, 1])

    @property
    def _rays_output(self) -> typ.Optional[Rays]:
        return self.rays_input

    @property
    def pre_transform(self) -> coordinate.TransformList:
        return coordinate.TransformList([self.transform])

    @property
    def post_transform(self) -> coordinate.TransformList:
        return coordinate.TransformList([coordinate.Translate(z=self.thickness)])

    def copy(self) -> 'CoordinateTransform':
        return CoordinateTransform(
            name=self.name.copy(),
            thickness=self.thickness.copy(),
            is_active=self.is_active.copy(),
            is_visible=self.is_visible.copy(),
            transform=self.transform.copy(),
        )

    def plot_2d(
            self,
            ax: plt.Axes,
            transform: typ.Optional[coordinate.Transform] = None,
            components: typ.Tuple[int, int] = (0, 1),
    ) -> plt.Axes:
        pass

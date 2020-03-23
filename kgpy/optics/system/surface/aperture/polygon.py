import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

from kgpy.optics.system.surface.aperture import Aperture, obscurable, decenterable

__all__ = ['Polygon']


@dataclasses.dataclass
class Polygon(decenterable.Decenterable, obscurable.Obscurable, Aperture):

    points: typ.Optional[u.Quantity] = None

    @property
    def config_broadcast(self):
        a = super().config_broadcast

        if self.points is not None:
            a = np.broadcast(a, self.points[..., 0, 0, 0])

        return a

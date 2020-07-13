import typing as typ
import abc
import dataclasses
import numpy as np
import astropy.units as u
import shapely.geometry
import kgpy.vector
from kgpy.vector import x, y, z
from . import Aperture, obscurable, decenterable

__all__ = ['Polygon']


@dataclasses.dataclass
class Polygon(decenterable.Decenterable, obscurable.Obscurable, Aperture, abc.ABC):

    def to_zemax(self) -> 'Polygon':
        raise NotImplementedError

    @property
    def shapely_poly(self) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(self.vertices)

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        p = shapely.geometry.Point(points[kgpy.vector.xy])
        return self.shapely_poly.contains(p)

    @property
    def min(self) -> u.Quantity:
        return kgpy.vector.from_components(self.vertices[x].min(), self.vertices[y].min(), self.vertices[z].min())

    @property
    def max(self) -> u.Quantity:
        return kgpy.vector.from_components(self.vertices[x].max(), self.vertices[y].max(), self.vertices[z].max())

    @property
    @abc.abstractmethod
    def vertices(self) -> u.Quantity:
        pass

    @property
    def wire(self) -> u.Quantity:
        return self.shapely_poly.interpolate(np.linspace(0, 1, num=self.num_samples, endpoint=False))

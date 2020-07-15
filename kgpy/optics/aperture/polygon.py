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
        sh = points.shape
        points = points.reshape((-1, ) + points.shape[~0:])
        p = shapely.geometry.MultiPoint(points[0:2].to(self.vertices.unit).value)
        poly = shapely.geometry.Polygon(self.wire)
        is_inside = poly.contains(p)
        if not self.is_obscuration:
            return is_inside
        else:
            return ~is_inside

    # def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
    #     p = shapely.geometry.Point(points[kgpy.vector.xy].to(self.vertices.unit).value)
    #     return self.shapely_poly.contains(p)

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
        coords = self.shapely_poly.exterior
        wire_samples = np.linspace(0, 1, num=self.num_samples, endpoint=False)
        return u.Quantity([coords.interpolate(a, normalized=True) << self.vertices.unit for a in wire_samples])

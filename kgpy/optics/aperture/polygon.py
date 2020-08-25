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

    @property
    def shapely_poly(self) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(self.vertices)

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:

        c = np.zeros(points[x].shape, dtype=np.bool)

        for v in range(self.vertices.shape[~1]):
            vert_j = self.vertices[..., v - 1, :]
            vert_i = self.vertices[..., v, :]
            slope = (vert_j[y] - vert_i[y]) / (vert_j[x] - vert_i[x])
            condition_1 = (vert_i[y] > points[y]) != (vert_j[y] > points[y])
            condition_2 = points[x] < ((points[y] - vert_i[y]) / slope + vert_i[x])
            mask = condition_1 & condition_2
            c[mask] = ~c[mask]

        if not self.is_obscuration:
            return c
        else:
            return ~c

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

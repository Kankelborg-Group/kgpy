import typing as typ
import abc
import dataclasses
import numpy as np
import astropy.units as u
import shapely.geometry
from . import Aperture, obscurable, decenterable

__all__ = ['Polygon']


@dataclasses.dataclass
class Polygon(decenterable.Decenterable, obscurable.Obscurable, Aperture, abc.ABC):

    def to_zemax(self) -> 'Polygon':
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def vertices(self) -> u.Quantity:
        pass

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        p = shapely.geometry.Point(points[0:2])
        poly = shapely.geometry.Polygon(self.vertices)
        return poly.contains(p)

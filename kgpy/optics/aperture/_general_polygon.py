import typing as typ
import dataclasses
import astropy.units as u
from . import Polygon

__all__ = ['GeneralPolygon']


@dataclasses.dataclass
class GeneralPolygon(Polygon):
    vertices: u.Quantity = u.Quantity([])
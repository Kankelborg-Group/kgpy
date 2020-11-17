import typing as typ
import dataclasses
import astropy.units as u
from kgpy import vector
from . import Polygon

__all__ = ['IrregularPolygon']


@dataclasses.dataclass
class IrregularPolygon(Polygon):
    vertices: u.Quantity = None

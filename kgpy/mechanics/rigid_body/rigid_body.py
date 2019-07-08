
from typing import List, Tuple, Union
import astropy.units as u
from shapely.geometry import Polygon

from kgpy import math

__all__ = ['Rectangular', 'Cylindrical']


class Rectangular:

    def __init__(self, cs: math.geometry.CoordinateSystem, dx: Union[u.Quantity, Tuple[u.Quantity, u.Quantity]],
                 dy: Union[u.Quantity, Tuple[u.Quantity, u.Quantity]],
                 dz: Union[u.Quantity, Tuple[u.Quantity, u.Quantity]]):

        # Save coordinate system
        self.cs = cs

        # Save x limits
        if isinstance(dx, tuple):
            self.dx0 = dx[0]
            self.dx1 = dx[1]

        else:
            self.dx0 = dx
            self.dx1 = dx

        # Save y limits
        if isinstance(dy, tuple):
            self.dy0 = dy[0]
            self.dy1 = dy[1]

        else:
            self.dy0 = dy
            self.dy1 = dy

        # Save z limits
        if isinstance(dz, tuple):
            self.dz0 = dz[0]
            self.dz1 = dz[1]

        else:
            self.dz0 = dz
            self.dz1 = dz

    @property
    def dX0(self) -> math.geometry.Vector:
        return -(self.dx0 * self.cs.x_hat)

    @property
    def dX1(self) -> math.geometry.Vector:
        return self.dx1 * self.cs.x_hat

    @property
    def dY0(self) -> math.geometry.Vector:
        return -(self.dy0 * self.cs.y_hat)

    @property
    def dY1(self) -> math.geometry.Vector:
        return self.dy1 * self.cs.y_hat

    @property
    def dZ0(self) -> math.geometry.Vector:
        return -(self.dz0 * self.cs.z_hat)

    @property
    def dZ1(self) -> math.geometry.Vector:
        return self.dz1 * self.cs.z_hat


class Cylindrical:

    def __init__(self, cs: math.geometry.CoordinateSystem, dr: Union[u.Quantity, Tuple[u.Quantity, u.Quantity]],
                 dz: Union[u.Quantity, Tuple[u.Quantity, u.Quantity]]):
        # Save coordinate system
        self.cs = cs

        # Save radial limits
        if isinstance(dr, tuple):
            self.dr0 = dr[0]
            self.dr1 = dr[1]

        else:
            self.dr0 = 0 * u.mm
            self.dr1 = dr

        # Save z limits
        if isinstance(dz, tuple):
            self.dz0 = dz[0]
            self.dz1 = dz[1]

        else:
            self.dz0 = dz
            self.dz1 = dz

    @property
    def dZ0(self) -> math.geometry.Vector:
        return -(self.dz0 * self.cs.z_hat)

    @property
    def dZ1(self) -> math.geometry.Vector:
        return self.dz1 * self.cs.z_hat


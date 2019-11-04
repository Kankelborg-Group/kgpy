import dataclasses
import abc
import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u

from kgpy import math, optics

__all__ = ['Aperture', 'Circular', 'Rectangular', 'RegularOctagon', 'Spider']


@dataclasses.dataclass
class Aperture(abc.ABC):

    decenter_x: u.Quantity = 0 * u.mm
    decenter_y: u.Quantity = 0 * u.mm

    is_obscuration: tp.Union[bool, npt.Array[bool]] = False
    
    @property
    def broadcasted_attrs(self):
        return np.broadcast(
            self.decenter_x,
            self.decenter_y,
            self.is_obscuration,
        )

    @property
    @abc.abstractmethod
    def points(self) -> u.Quantity:
        pass


@dataclasses.dataclass
class Circular(Aperture):

    inner_radius: u.Quantity = 0 * u.mm
    outer_radius: u.Quantity = 0 * u.mm
    
    @property
    def broadcasted_attrs(self):
        return np.broadcast(
            super().broadcasted_attrs,
            self.inner_radius,
            self.outer_radius,
        )

    @property
    def points(self) -> u.Quantity:
        raise NotImplementedError


@dataclasses.dataclass
class Rectangular(Aperture):

    half_width_x: u.Quantity = 0 * u.mm
    half_width_y: u.Quantity = 0 * u.mm
    
    @property
    def broadcasted_attrs(self):
        return np.broadcast(
            super().broadcasted_attrs,
            self.half_width_x,
            self.half_width_y,
        )

    @property
    def points(self) -> u.Quantity:
        return u.Quantity([
            u.Quantity([self.half_width_x, self.half_width_y]),
            u.Quantity([self.half_width_x, -self.half_width_y]),
            u.Quantity([-self.half_width_x, -self.half_width_y]),
            u.Quantity([-self.half_width_x, self.half_width_y]),
        ])


@dataclasses.dataclass
class RegularPolygon(Aperture):

    radius: u.Quantity = 0 * u.mm
    num_sides: u.Quantity = 8

    @property
    def broadcasted_attrs(self):
        return np.broadcast(
            super().broadcasted_attrs,
            self.radius,
            self.num_sides,
        )

    @property
    def points(self) -> u.Quantity:

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, self.num_sides, endpoint=False) * u.rad  # type: u.Quantity

        # Calculate points
        x = self.radius * np.cos(angles)  # type: u.Quantity
        y = self.radius * np.sin(angles)  # type: u.Quantity
        pts = u.Quantity([x, y])
        pts = pts.transpose()   # type: u.Quantity

        return pts


class RegularOctagon(RegularPolygon):

    def __init__(self, radius: u.Quantity = 0 * u.mm, **kwargs):

        super().__init__(radius=radius, num_sides=8, **kwargs)


@dataclasses.dataclass
class Polygon(Aperture):
    
    points: u.Quantity = [[-1, -1], [-1, 1], [1, 1], [1, -1]] * u.mm

    @property
    def broadcasted_attrs(self):
        return np.broadcast(
            super().broadcasted_attrs,
            self.points[..., 0, 0],
        )


@dataclasses.dataclass
class Spider(Aperture):

    arm_half_width: u.Quantity = 0 * u.mm
    num_arms: int = 2

    @property
    def broadcasted_attrs(self):
        return np.broadcast(
            super().broadcasted_attrs,
            self.arm_half_width,
            self.num_arms
        )

    @property
    def points(self) -> u.Quantity:
        raise NotImplementedError

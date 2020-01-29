import dataclasses
import abc
import typing as tp
import numpy as np
import astropy.units as u

from .. import mixin

__all__ = ['Aperture', 'Circular', 'Rectangular', 'RegularOctagon', 'Spider', 'UserPolygon']


@dataclasses.dataclass
class Aperture(mixin.ConfigBroadcast):
    pass


class NoAperture(Aperture):
    pass


@dataclasses.dataclass
class Decenterable(mixin.ConfigBroadcast):

    decenter_x: u.Quantity = 0 * u.mm
    decenter_y: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.decenter_x,
            self.decenter_y,
        )


@dataclasses.dataclass
class Obscurable(mixin.ConfigBroadcast):

    is_obscuration: np.ndarray[bool] = dataclasses.field(default_factory=lambda: np.array(False))

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.is_obscuration,
        )


class Polygon(abc.ABC, Aperture):

    @property
    @abc.abstractmethod
    def points(self):
        ...


@dataclasses.dataclass
class Circular(Obscurable, Decenterable, Polygon):

    inner_radius: u.Quantity = 0 * u.mm
    outer_radius: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.inner_radius,
            self.outer_radius,
        )

    @property
    def points(self) -> u.Quantity:
        a = np.linspace(0 * u.deg, 360 * u.deg, num=100)

        x = u.Quantity([np.cos(a), np.sin(a)])
        x = np.moveaxis(x, 0, ~0)

        return u.Quantity([self.outer_radius * x, self.inner_radius * x])


@dataclasses.dataclass
class Rectangular(Obscurable, Decenterable, Polygon):

    half_width_x: u.Quantity = 0 * u.mm
    half_width_y: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.half_width_x,
            self.half_width_y,
        )

    @property
    def points(self) -> u.Quantity:
        return u.Quantity([u.Quantity([
            u.Quantity([self.half_width_x, self.half_width_y]),
            u.Quantity([self.half_width_x, -self.half_width_y]),
            u.Quantity([-self.half_width_x, -self.half_width_y]),
            u.Quantity([-self.half_width_x, self.half_width_y]),
        ])])


@dataclasses.dataclass
class RegularPolygon(Aperture):

    radius: u.Quantity = 0 * u.mm
    num_sides: int = 8

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
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
        pts = u.Quantity(pts.transpose())
        pts = u.Quantity([pts])

        return pts


class RegularOctagon(RegularPolygon):

    def __init__(self, radius: u.Quantity = 0 * u.mm, **kwargs):
        super().__init__(radius=radius, num_sides=8, **kwargs)


@dataclasses.dataclass
class UserPolygonMixin(mixin.ConfigBroadcast, Aperture):

    points: u.Quantity = dataclasses.field(default_factory=lambda: u.Quantity([]))

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.points[..., 0, 0, 0],
        )


@dataclasses.dataclass
class UserPolygon(Obscurable, Decenterable, UserPolygonMixin):
    pass


@dataclasses.dataclass
class Spider(Decenterable, Polygon):

    arm_half_width: u.Quantity = 0 * u.mm
    num_arms: int = 2
    radius: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.arm_half_width,
            self.num_arms,
        )

    @property
    def points(self) -> u.Quantity:
        a = np.linspace(0 * u.deg, 360 * u.deg, self.num_arms, endpoint=False)
        a = np.expand_dims(a, ~0)

        x = u.Quantity([0 * u.m, self.radius, self.radius, 0 * u.m])
        y = u.Quantity([-self.arm_half_width, -self.arm_half_width, self.arm_half_width, self.arm_half_width])

        xp = x * np.cos(a) - y * np.sin(a)
        yp = x * np.sin(a) + y * np.cos(a)

        pts = u.Quantity([xp, yp])
        pts = np.moveaxis(pts, 0, ~0)
        return pts

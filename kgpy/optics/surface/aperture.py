"""
Aperture models of various shapes.
"""
import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import shapely.geometry
from kgpy import mixin, vector, transform
from .sag import Sag

__all__ = [
    'Aperture',
    'Decenterable',
    'Obscurable',
    'Circular',
    'Polygon',
    'RegularPolygon',
    'IrregularPolygon',
    'Rectangular',
    'AsymmetricRectangular',
    'IsoscelesTrapezoid',
]


@dataclasses.dataclass
class Aperture(
    mixin.Broadcastable,
    mixin.Copyable,
    abc.ABC
):
    num_samples: int = 1000

    @abc.abstractmethod
    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def min(self) -> u.Quantity:
        pass

    @property
    @abc.abstractmethod
    def max(self) -> u.Quantity:
        pass

    @property
    @abc.abstractmethod
    def wire(self) -> u.Quantity:
        pass

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[Sag] = None,
            color: str = 'black'
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        with astropy.visualization.quantity_support():
            c1, c2 = components
            wire = self.wire
            if wire.unit.is_equivalent(u.mm):
                if sag is not None:
                    wire[vector.z] = wire[vector.z] + sag(wire[vector.x], wire[vector.y])
                if rigid_transform is not None:
                    wire = rigid_transform(wire, num_extra_dims=1)
                wire = wire.reshape((-1, ) + wire.shape[~1:])
                ax.fill(wire[..., c1].T, wire[..., c2].T, color=color, fill=False)
        return ax

    def copy(self) -> 'Aperture':
        other = super().copy()      # type: Aperture
        other.num_samples = self.num_samples
        return other


@dataclasses.dataclass
class Decenterable(
    mixin.Broadcastable,
    mixin.Copyable,
):

    decenter: transform.rigid.Translate = dataclasses.field(default_factory=lambda: transform.rigid.Translate())

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.decenter.broadcasted)
        return out

    def copy(self) -> 'Decenterable':
        other = super().copy()
        other.decenter = self.decenter.copy()
        return other


@dataclasses.dataclass
class Obscurable(
    mixin.Copyable,
):
    is_obscuration: bool = False

    def copy(self) -> 'Obscurable':
        other = super().copy()      # type: Obscurable
        other.is_obscuration = self.is_obscuration
        return other


@dataclasses.dataclass
class Circular(
    Aperture,
    Decenterable,
    Obscurable,
):

    radius: u.Quantity = 0 * u.mm

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.radius)
        return out

    @property
    def min(self) -> u.Quantity:
        return -vector.from_components(self.radius, self.radius) + self.decenter.vector

    @property
    def max(self) -> u.Quantity:
        return vector.from_components(self.radius, self.radius) + self.decenter.vector

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        x = points[..., 0] - self.decenter.x
        y = points[..., 1] - self.decenter.y
        r = np.sqrt(np.square(x) + np.square(y))
        is_inside = r <= self.radius
        if not self.is_obscuration:
            return is_inside
        else:
            return ~is_inside

    @property
    def wire(self) -> u.Quantity:

        a = np.linspace(0 * u.deg, 360 * u.deg, num=self.num_samples)
        r = np.expand_dims(self.radius.copy(), ~0)

        x = r * np.cos(a) + self.decenter.x
        y = r * np.sin(a) + self.decenter.y
        z = np.broadcast_to(0, x.shape)

        return np.stack([x, y, z], axis=~0)

    def copy(self) -> 'Circular':
        other = super().copy()      # type: Circular
        other.radius = self.radius.copy()
        return other


@dataclasses.dataclass
class Polygon(Decenterable, Obscurable, Aperture, abc.ABC):

    @property
    def shapely_poly(self) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(self.vertices)

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:

        points -= self.decenter.vector

        x, y = vector.x, vector.y

        c = np.zeros(points[x].shape, dtype=np.bool)

        for v in range(self.vertices.shape[~1]):
            vertices = self.vertices[..., None, None, None, None, None, :, :]
            vert_j = vertices[..., v - 1, :]
            vert_i = vertices[..., v, :]
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
        return vector.from_components(
            self.vertices[vector.x].min(), self.vertices[vector.y].min(), self.vertices[vector.z].min())

    @property
    def max(self) -> u.Quantity:
        return vector.from_components(
            self.vertices[vector.x].max(), self.vertices[vector.y].max(), self.vertices[vector.z].max())

    @property
    @abc.abstractmethod
    def vertices(self) -> u.Quantity:
        pass

    @property
    def wire(self) -> u.Quantity:
        left_vert = np.roll(self.vertices[..., None, :], -1, axis=~2)
        right_vert = self.vertices[..., None, :]
        diff = left_vert - right_vert
        t = np.linspace(0, 1, num=self.num_samples, endpoint=False)[..., None, :, None]
        wire = right_vert + diff * t
        wire = wire.reshape(wire.shape[:~2] + (wire.shape[~2] * wire.shape[~1], ) + wire.shape[~0:])
        return wire


@dataclasses.dataclass
class RegularPolygon(Polygon):

    radius: u.Quantity = 0 * u.mm
    num_sides: int = 8
    offset_angle: u.Quantity = 0 * u.deg

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.radius)
        return out

    @property
    def vertices(self) -> u.Quantity:
        angles = np.linspace(self.offset_angle, 360 * u.deg + self.offset_angle, self.num_sides, endpoint=False)
        return vector.from_components_cylindrical(self.radius, angles)

    @property
    def edge_subtent(self):
        """
        Calculate the angle subtended by each edge of the polygon by dividing the angle of a circle (360 degrees) by
        the number of sides in the regular polygon.
        :return: Angle subtended by each edge
        """
        return 360 * u.deg / self.num_sides

    @property
    def half_edge_subtent(self):
        """
        Calculate the angle subtended between a vertex and a point on the center of an edge.
        This is sometimes a more useful quantity than the subtent of an entire edge.
        :return:
        """
        return self.edge_subtent / 2

    @property
    def min_radius(self):
        """
        Calculate the distance from the center of the polygon to the center of an edge of a polygon.
        :return: The minimum radius of the polygon.
        """
        return self.radius * np.cos(self.half_edge_subtent)

    def copy(self) -> 'RegularPolygon':
        other = super().copy()      # type: RegularPolygon
        other.radius = self.radius.copy()
        other.num_sides = self.num_sides
        other.offset_angle = self.offset_angle.copy()
        return other


@dataclasses.dataclass
class IrregularPolygon(Polygon):
    vertices: u.Quantity = None


@dataclasses.dataclass
class Rectangular(Polygon):

    half_width_x: u.Quantity = 0 * u.mm
    half_width_y: u.Quantity = 0 * u.mm

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.half_width_x)
        out = np.broadcast(out, self.half_width_y)
        return out

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        amin = self.min[vector.xy][..., None, None, None, None, None, :]
        amax = self.max[vector.xy][..., None, None, None, None, None, :]
        x = points[vector.x]
        y = points[vector.y]
        m1 = x <= amax[vector.x]
        m2 = x >= amin[vector.x]
        m3 = y <= amax[vector.y]
        m4 = y >= amin[vector.y]
        is_inside = m1 & m2 & m3 & m4
        if not self.is_obscuration:
            return is_inside
        else:
            return ~is_inside

    @property
    def min(self) -> u.Quantity:
        return -vector.from_components(self.half_width_x, self.half_width_y) + self.decenter.vector

    @property
    def max(self) -> u.Quantity:
        return vector.from_components(self.half_width_x, self.half_width_y) + self.decenter.vector

    @property
    def vertices(self) -> u.Quantity:

        minx, miny = self.min[vector.x], self.min[vector.y]
        maxx, maxy = self.max[vector.x], self.max[vector.y]

        x = np.stack([maxx, maxx, minx, minx], axis=~0)
        y = np.stack([maxy, miny, miny, maxy], axis=~0)

        return vector.from_components(x, y)

    # @property
    # def wire(self) -> u.Quantity:
    #
    #     wx, wy = np.broadcast_arrays(self.half_width_x, self.half_width_y, subok=True)
    #
    #     rx = np.linspace(-wx, wx, self.num_samples, axis=~0)
    #     ry = np.linspace(-wy, wy, self.num_samples, axis=~0)
    #
    #     wx = np.expand_dims(wx, ~0)
    #     wy = np.expand_dims(wy, ~0)
    #
    #     wx, rx = np.broadcast_arrays(wx, rx, subok=True)
    #     wy, ry = np.broadcast_arrays(wy, ry, subok=True)
    #
    #     x = np.stack([rx, wx, rx[::-1], -wx])
    #     y = np.stack([wy, ry[::-1], -wy, ry])
    #
    #     return kgpy.vector.from_components(x, y)

    def copy(self) -> 'Rectangular':
        other = super().copy()      # type: Rectangular
        other.half_width_x = self.half_width_x.copy()
        other.half_width_y = self.half_width_y.copy()
        return other


@dataclasses.dataclass
class AsymmetricRectangular(Polygon):

    width_x_neg: u.Quantity = 0 * u.mm
    width_x_pos: u.Quantity = 0 * u.mm
    width_y_neg: u.Quantity = 0 * u.mm
    width_y_pos: u.Quantity = 0 * u.mm

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.width_x_neg)
        out = np.broadcast(out, self.width_x_pos)
        out = np.broadcast(out, self.width_y_neg)
        out = np.broadcast(out, self.width_y_pos)
        return out

    @property
    def vertices(self) -> u.Quantity:
        v_x = np.stack([self.width_x_pos, self.width_x_neg, self.width_x_neg, self.width_x_pos]) + self.decenter.x
        v_y = np.stack([self.width_y_pos, self.width_y_pos, self.width_y_neg, self.width_y_neg]) + self.decenter.y
        return vector.from_components(v_x, v_y)

    def copy(self) -> 'AsymmetricRectangular':
        other = super().copy()  # type: AsymmetricRectangular
        other.width_x_neg = self.width_x_neg.copy()
        other.width_x_pos = self.width_x_pos.copy()
        other.width_y_neg = self.width_y_neg.copy()
        other.width_y_pos = self.width_y_pos.copy()
        return other


@dataclasses.dataclass
class IsoscelesTrapezoid(Polygon):
    apex_offset: u.Quantity = 0 * u.mm
    half_width_left: u.Quantity = 0 * u.mm
    half_width_right: u.Quantity = 0 * u.mm
    wedge_half_angle: u.Quantity = 0 * u.deg

    @property
    def vertices(self) -> u.Quantity:
        m = np.tan(self.wedge_half_angle)
        # inner_radius = self.half_width_left +
        # inner_radius = self.apex_offset - self.half_width_left
        # outer_radius = self.apex_offset + self.half_width_right
        left_x, left_y = -self.half_width_left, -m * (self.apex_offset + self.half_width_left)
        right_x, right_y = self.half_width_right, -m * (self.apex_offset - self.half_width_right)
        v_x = np.stack([left_x, right_x, right_x, left_x], axis=~0)
        v_y = np.stack([left_y, right_y, -right_y, -left_y], axis=~0)
        return vector.from_components(v_x, v_y)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.apex_offset)
        out = np.broadcast(out, self.half_width_left)
        out = np.broadcast(out, self.half_width_right)
        out = np.broadcast(out, self.wedge_half_angle)
        return out

    def copy(self) -> 'IsoscelesTrapezoid':
        other = super().copy()      # type: IsoscelesTrapezoid
        other.apex_offset = self.apex_offset.copy()
        other.half_width_left = self.half_width_left.copy()
        other.half_width_right = self.half_width_right.copy()
        other.wedge_half_angle = self.wedge_half_angle.copy()
        return other

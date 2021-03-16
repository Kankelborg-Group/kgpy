"""
Aperture models of various shapes.
"""
import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.axes
import matplotlib.lines
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
    mixin.Colorable,
    abc.ABC
):
    num_samples: int = 1000

    @abc.abstractmethod
    def is_unvignetted(self, points: vector.Vector2D, num_extra_dims: int = 0) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def min(self) -> vector.Vector3D:
        pass

    @property
    @abc.abstractmethod
    def max(self) -> vector.Vector3D:
        pass

    @property
    @abc.abstractmethod
    def wire(self) -> vector.Vector3D:
        pass

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            components: typ.Tuple[str, str] = ('x', 'y'),
            component_z: typ.Optional[str] = None,
            color: typ.Optional[str] = None,
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[Sag] = None,
    ) -> typ.List[matplotlib.lines.Line2D]:

        if color is None:
            color = self.color

        with astropy.visualization.quantity_support():
            c1, c2 = components
            wire = self.wire
            lines = []
            if wire.x.unit.is_equivalent(u.mm):
                if sag is not None:
                    wire.z = wire.z + sag(wire.x, wire.y)
                if transform_extra is not None:
                    wire = transform_extra(wire, num_extra_dims=1)
                wire = wire.reshape((-1, wire.shape[~0]))
                plot_kwargs_z = {}

                for i in range(wire.shape[0]):
                    if component_z is not None:
                        plot_kwargs_z['zs'] = wire[i].get_component(component_z)
                    lines += ax.plot(
                        wire[i].get_component(c1),
                        wire[i].get_component(c2),
                        color=color,
                        **plot_kwargs_z
                    )

        return lines

    def view(self) -> 'Aperture':
        other = super().view()  # type: Aperture
        other.num_samples = self.num_samples
        return other

    def copy(self) -> 'Aperture':
        other = super().copy()  # type: Aperture
        other.num_samples = self.num_samples
        return other


@dataclasses.dataclass
class Decenterable(
    mixin.Broadcastable,
    mixin.Copyable,
):
    decenter: transform.rigid.Translate = dataclasses.field(default_factory=transform.rigid.Translate)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.decenter.broadcasted)
        return out

    def view(self) -> 'Decenterable':
        other = super().view()  # type: Decenterable
        other.decenter = self.decenter
        return other

    def copy(self) -> 'Decenterable':
        other = super().copy()
        other.decenter = self.decenter.copy()
        return other


@dataclasses.dataclass
class Obscurable(
    mixin.Copyable,
):
    is_obscuration: bool = False

    def view(self) -> 'Obscurable':
        other = super().view()  # type: Obscurable
        other.is_obscuration = self.is_obscuration
        return other

    def copy(self) -> 'Obscurable':
        other = super().copy()  # type: Obscurable
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
    def min(self) -> vector.Vector3D:
        return -vector.Vector3D(x=self.radius, y=self.radius, z=0 * self.radius.unit) + self.decenter.value

    @property
    def max(self) -> vector.Vector3D:
        return vector.Vector3D(x=self.radius, y=self.radius, z=0 * self.radius.unit) + self.decenter.value

    def is_unvignetted(self, points: vector.Vector2D, num_extra_dims: int = 0) -> np.ndarray:
        extra_dim_slice = (Ellipsis, ) + num_extra_dims * (np.newaxis, )
        x = points.x - self.decenter.x[extra_dim_slice]
        y = points.y - self.decenter.y[extra_dim_slice]
        r = np.sqrt(np.square(x) + np.square(y))
        is_inside = r <= self.radius[extra_dim_slice]
        if not self.is_obscuration:
            return is_inside
        else:
            return ~is_inside

    @property
    def wire(self) -> vector.Vector3D:
        wire = vector.Vector3D.from_cylindrical(
            radius=self.radius[..., np.newaxis],
            azimuth=np.linspace(0 * u.deg, 360 * u.deg, num=self.num_samples),
            z=0 * self.radius,
        )
        return wire + self.decenter.value[..., np.newaxis, np.newaxis]

    def view(self) -> 'Circular':
        other = super().view()  # type: Circular
        other.radius = self.radius
        return other

    def copy(self) -> 'Circular':
        other = super().copy()  # type: Circular
        other.radius = self.radius.copy()
        return other


@dataclasses.dataclass
class Polygon(Decenterable, Obscurable, Aperture, abc.ABC):

    @property
    def shapely_poly(self) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(self.vertices)

    def is_unvignetted(self, points: vector.Vector2D, num_extra_dims: int = 0) -> np.ndarray:

        points = points - self.decenter.value

        extra_dims_slice = (Ellipsis, ) + num_extra_dims * (np.newaxis, ) + (slice(None), )
        vertices = self.vertices[extra_dims_slice]

        c = np.zeros(points.shape, dtype=np.bool)

        for v in range(vertices.shape[~0]):
            vert_j = vertices[..., v - 1]
            vert_i = vertices[..., v]
            slope = (vert_j.y - vert_i.y) / (vert_j.x - vert_i.x)
            condition_1 = (vert_i.y > points.y) != (vert_j.y > points.y)
            condition_2 = points.x < ((points.y - vert_i.y) / slope + vert_i.x)
            mask = condition_1 & condition_2
            c[mask] = ~c[mask]

        if not self.is_obscuration:
            return c
        else:
            return ~c

    @property
    def min(self) -> vector.Vector3D:
        return self.vertices.min()

    @property
    def max(self) -> vector.Vector3D:
        return self.vertices.max()

    @property
    @abc.abstractmethod
    def vertices(self) -> vector.Vector3D:
        pass

    @property
    def wire(self) -> u.Quantity:
        left_vert = np.roll(self.vertices, -1, axis=~0)[..., np.newaxis]
        right_vert = self.vertices[..., np.newaxis]
        diff = left_vert - right_vert
        t = np.linspace(0, 1, num=self.num_samples, endpoint=False)[..., np.newaxis, :]
        wire = right_vert + diff * t
        wire = wire.reshape(wire.shape[:~1] + (-1,))
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
    def vertices(self) -> vector.Vector3D:
        vertices = vector.Vector3D.from_cylindrical(
            radius=self.radius,
            azimuth=np.linspace(self.offset_angle, 360 * u.deg + self.offset_angle, self.num_sides, endpoint=False),
            z=0 * self.radius,
        )
        return vertices + self.decenter.value

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

    def view(self) -> 'RegularPolygon':
        other = super().view()  # type: RegularPolygon
        other.radius = self.radius
        other.num_sides = self.num_sides
        other.offset_angle = self.offset_angle
        return other

    def copy(self) -> 'RegularPolygon':
        other = super().copy()  # type: RegularPolygon
        other.radius = self.radius.copy()
        other.num_sides = self.num_sides
        other.offset_angle = self.offset_angle.copy()
        return other


@dataclasses.dataclass
class IrregularPolygon(Polygon):
    vertices: vector.Vector3D = None

    def view(self) -> 'IrregularPolygon':
        other = super().view()  # type: IrregularPolygon
        other.vertices = self.vertices
        return other

    def copy(self) -> 'IrregularPolygon':
        other = super().copy()  # type: IrregularPolygon
        if self.vertices is not None:
            other.vertices = self.vertices.copy()
        else:
            other.vertices = self.vertices
        return other


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

    def is_unvignetted(self, points: vector.Vector2D, num_extra_dims: int = 0) -> np.ndarray:
        extra_dims_slice = (Ellipsis, ) + num_extra_dims * (np.newaxis, )
        amin = self.min[extra_dims_slice]
        amax = self.max[extra_dims_slice]
        m1 = points.x <= amax.x
        m2 = points.x >= amin.x
        m3 = points.y <= amax.y
        m4 = points.y >= amin.y
        is_inside = m1 & m2 & m3 & m4
        if not self.is_obscuration:
            return is_inside
        else:
            return ~is_inside

    @property
    def min(self) -> vector.Vector3D:
        return -vector.Vector3D(
            x=self.half_width_x, y=self.half_width_y, z=0 * self.half_width_x.unit) + self.decenter.value

    @property
    def max(self) -> vector.Vector3D:
        return vector.Vector3D(
            x=self.half_width_x, y=self.half_width_y, z=0 * self.half_width_x.unit) + self.decenter.value

    @property
    def vertices(self) -> vector.Vector3D:

        minx, miny = self.min.x, self.min.y
        maxx, maxy = self.max.x, self.max.y

        return vector.Vector3D(
            x=np.stack([maxx, maxx, minx, minx], axis=~0),
            y=np.stack([maxy, miny, miny, maxy], axis=~0),
            z=0 * minx,
        )

    def view(self) -> 'Rectangular':
        other = super().view()  # type: Rectangular
        other.half_width_x = self.half_width_x
        other.half_width_y = self.half_width_y
        return other

    def copy(self) -> 'Rectangular':
        other = super().copy()  # type: Rectangular
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
    def vertices(self) -> vector.Vector3D:
        vertices = vector.Vector3D(
            x=np.stack([self.width_x_pos, self.width_x_neg, self.width_x_neg, self.width_x_pos], axis=~0),
            y=np.stack([self.width_y_pos, self.width_y_pos, self.width_y_neg, self.width_y_neg], axis=~0),
            z=0 * self.width_x_pos,
        )
        return vertices + self.decenter.value

    def view(self) -> 'AsymmetricRectangular':
        other = super().view()  # type: AsymmetricRectangular
        other.width_x_neg = self.width_x_neg
        other.width_x_pos = self.width_x_pos
        other.width_y_neg = self.width_y_neg
        other.width_y_pos = self.width_y_pos
        return other

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
    def vertices(self) -> vector.Vector3D:
        m = np.tan(self.wedge_half_angle)
        left_x, left_y = -self.half_width_left, -m * (self.apex_offset + self.half_width_left)
        right_x, right_y = self.half_width_right, -m * (self.apex_offset - self.half_width_right)
        vertices = vector.Vector3D(
            x=np.stack([left_x, right_x, right_x, left_x], axis=~0),
            y=np.stack([left_y, right_y, -right_y, -left_y], axis=~0),
            z=0 * left_x,
        )
        return vertices + self.decenter.value

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.apex_offset)
        out = np.broadcast(out, self.half_width_left)
        out = np.broadcast(out, self.half_width_right)
        out = np.broadcast(out, self.wedge_half_angle)
        return out

    def view(self) -> 'IsoscelesTrapezoid':
        other = super().view()  # type: IsoscelesTrapezoid
        other.apex_offset = self.apex_offset
        other.half_width_left = self.half_width_left
        other.half_width_right = self.half_width_right
        other.wedge_half_angle = self.wedge_half_angle
        return other

    def copy(self) -> 'IsoscelesTrapezoid':
        other = super().copy()  # type: IsoscelesTrapezoid
        other.apex_offset = self.apex_offset.copy()
        other.half_width_left = self.half_width_left.copy()
        other.half_width_right = self.half_width_right.copy()
        other.wedge_half_angle = self.wedge_half_angle.copy()
        return other

"""
Aperture models of various shapes.
"""
import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.axes
import matplotlib.lines
import mpl_toolkits.mplot3d.art3d
import astropy.units as u
import astropy.visualization
import shapely.geometry
from ezdxf.addons.r12writer import R12FastStreamWriter
import kgpy.mixin
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.transforms
import kgpy.io.dxf
from . import sags

__all__ = [
    'Aperture',
    # 'Decenterable',
    'Obscurable',
    'Circular',
    'Polygon',
    'RegularPolygon',
    'IrregularPolygon',
    'Rectangular',
    # 'AsymmetricRectangular',
    'IsoscelesTrapezoid',
]

ApertureT = typ.TypeVar('ApertureT', bound='Aperture')
DecenterableT = typ.TypeVar('DecenterableT', bound='Decenterable')
ObscurableT = typ.TypeVar('ObscurableT', bound='Obscurable')
CircularT = typ.TypeVar('CircularT', bound='Circular')
PolygonT = typ.TypeVar('PolygonT', bound='Polygon')
RegularPolygonT = typ.TypeVar('RegularPolygonT', bound='RegularPolygon')
IrregularPolygonT = typ.TypeVar('IrregularPolygonT', bound='IrregularPolygon')
RectangularT = typ.TypeVar('RectangularT', bound='Rectangular')
AsymmetricRectangularT = typ.TypeVar('AsymmetricRectangularT', bound='AsymmetricRectangular')
IsoscelesTrapezoidT = typ.TypeVar('IsoscelesTrapezoidT', bound='IsoscelesTrapezoid')


@dataclasses.dataclass
class Aperture(
    kgpy.io.dxf.WritableMixin,
    kgpy.transforms.Transformable,
    kgpy.mixin.Broadcastable,
    kgpy.mixin.Plottable,
    abc.ABC
):
    num_samples: int = 1000

    @abc.abstractmethod
    def is_unvignetted(self: ApertureT, position: kgpy.vectors.Cartesian2D) -> kgpy.uncertainty.ArrayLike:
        pass

    @property
    @abc.abstractmethod
    def min(self: ApertureT) -> kgpy.vectors.Cartesian3D:
        pass

    @property
    @abc.abstractmethod
    def max(self: ApertureT) -> kgpy.vectors.Cartesian3D:
        pass

    @property
    def vertices(self: ApertureT) -> typ.Optional[kgpy.vectors.Cartesian3D]:
        return None

    @property
    @abc.abstractmethod
    def wire(self: ApertureT) -> kgpy.vectors.Cartesian3D:
        pass

    def plot(
            self: ApertureT,
            ax: matplotlib.axes.Axes,
            component_x: str = 'x',
            component_y: str = 'y',
            component_z: typ.Optional[str] = None,
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            sag: typ.Optional[sags.Sag] = None,
            **kwargs: typ.Any
    ) -> typ.List[matplotlib.lines.Line2D]:

        kwargs = {**self.plot_kwargs, **kwargs}

        with astropy.visualization.quantity_support():
            wire = self.wire
            wire = np.broadcast_to(wire, wire.shape)
            lines = []
            if wire.x.unit.is_equivalent(u.mm):
                if sag is not None:
                    wire.z = sag(wire)
                if transform_extra is not None:
                    wire = transform_extra(wire)

                # shape = wire.shape
                # shape.pop('wire')
                # for key in plot_kwargs:
                #     shape = np.broadcast_shapes(shape, np.array(plot_kwargs[key]).shape,)
                #
                # wire = np.broadcast_to(wire, shape + wire.shape[~0:], subok=True)

                # for key in plot_kwargs:
                #     if not isinstance(plot_kwargs[key], kgpy.labeled.ArrayInterface):
                #         plot_kwargs[key] = kgpy.labeled.Array(plot_kwargs[key])
                #     plot_kwargs[key] = np.broadcast_to(plot_kwargs[key], shape)

                # linestyle = np.broadcast_to(np.array(linestyle), shape).reshape(-1)

                # wire = wire.reshape((-1, wire.shape[~0]))

                if isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
                    return wire.plot_filled(
                        ax=ax,
                        axis_plot='wire',
                        component_x=component_x,
                        component_y=component_y,
                        component_z=component_z,
                        **kwargs,
                    )

                else:
                    return wire.plot(
                        ax=ax,
                        axis_plot='wire',
                        component_x=component_x,
                        component_y=component_y,
                        component_z=component_z,
                        **kwargs,
                    )




                # for index in wire.ndindex(axis_ignored='vertex'):
                # # for i in range(wire.shape[0]):
                #     plot_kwargs_index = {k: plot_kwargs[k][index] for k in plot_kwargs}
                #
                #     if component_z is not None:
                #
                #         if 'color' in plot_kwargs_index:
                #             plot_kwargs_index['edgecolors'] = plot_kwargs_index.pop('color')
                #         if 'linewidth' in plot_kwargs_index:
                #             plot_kwargs_index['linewidths'] = plot_kwargs_index.pop('linewidth')
                #         if 'linestyle' in plot_kwargs_index:
                #             plot_kwargs_index['linestyles'] = plot_kwargs_index.pop('linestyle')
                #
                #         lines += [ax.add_collection(mpl_toolkits.mplot3d.art3d.Poly3DCollection(
                #             verts=[np.stack([
                #                 wire[index].get_component(components[0]),
                #                 wire[index].get_component(components[1]),
                #                 wire[index].get_component(component_z),
                #             ], axis=~0)],
                #             facecolors='white',
                #             **plot_kwargs_index
                #         ))]
                #
                #     else:
                #         lines += ax.plot(
                #             wire[index].get_component(c1),
                #             wire[index].get_component(c2),
                #             **plot_kwargs_index,
                #         )

        return lines

    def write_to_dxf(
            self: ApertureT,
            file_writer: R12FastStreamWriter,
            unit: u.Unit,
            transform_extra: typ.Optional[kgpy.transforms.AbstractTransform] = None,
            sag: typ.Optional[sags.Sag] = None
    ) -> None:

        super().write_to_dxf(
            file_writer=file_writer,
            unit=unit,
            transform_extra=transform_extra,
        )

        wire = self.wire
        if wire.x.unit.is_equivalent(unit):
            if sag is not None:
                wire.z = wire.z + sag(wire.xy)
            if transform_extra is not None:
                wire = transform_extra(wire)

            for index in wire.ndindex(axis_ignored='wire'):

                coords = wire[index].coordinates

                file_writer.add_polyline(
                    vertices=np.stack([coords[c].array for c in coords], axis=~0).to(unit).value
                )


@dataclasses.dataclass
class Obscurable(
    kgpy.mixin.Copyable,
):
    is_obscuration: kgpy.labeled.ArrayLike = False


@dataclasses.dataclass
class Circular(
    Aperture,
    Obscurable,
):
    radius: kgpy.uncertainty.ArrayLike = 0 * u.mm

    @property
    def broadcasted(self: CircularT):
        out = super().broadcasted
        out = np.broadcast(out, self.radius)
        return out

    @property
    def min(self: CircularT) -> kgpy.vectors.Cartesian3D:
        return -kgpy.vectors.Cartesian3D(x=self.radius, y=self.radius, z=0 * self.radius) + self.decenter.vector

    @property
    def max(self: CircularT) -> kgpy.vectors.Cartesian3D:
        return kgpy.vectors.Cartesian3D(x=self.radius, y=self.radius, z=0 * self.radius) + self.decenter.vector

    def is_unvignetted(self: CircularT, position: kgpy.vectors.Cartesian2D) -> kgpy.uncertainty.ArrayLike:
        position = self.transform.inverse(position.to_3d())
        is_inside = position.length <= self.radius
        if not self.is_obscuration:
            return is_inside
        else:
            return ~is_inside

    @property
    def wire(self: CircularT) -> kgpy.vectors.Cartesian3D:
        wire = kgpy.vectors.Cylindrical(
            radius=self.radius[..., np.newaxis],
            azimuth=kgpy.labeled.LinearSpace(0 * u.deg, 360 * u.deg, num=self.num_samples, axis='wire'),
            z=0 * self.radius,
        ).cartesian
        return self.transform(wire)


@dataclasses.dataclass
class Polygon(
    Obscurable,
    Aperture,
    abc.ABC,
):

    @property
    def shapely_poly(self: PolygonT) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(self.vertices)

    def is_unvignetted(self: PolygonT, position: kgpy.vectors.Cartesian2D) -> kgpy.uncertainty.ArrayLike:

        # position = self.transform(position)

        vertices = self.vertices

        result = False
        for v in range(vertices.shape['vertex']):
            vert_j = vertices[dict(vertex=v - 1)]
            vert_i = vertices[dict(vertex=v)]
            slope = (vert_j.y - vert_i.y) / (vert_j.x - vert_i.x)
            condition_1 = (vert_i.y > position.y) != (vert_j.y > position.y)
            condition_2 = position.x < ((position.y - vert_i.y) / slope + vert_i.x)
            result = result ^ (condition_1 & condition_2)

        if not self.is_obscuration:
            return result
        else:
            return ~result

    @property
    def min(self: PolygonT) -> kgpy.vectors.Cartesian3D:
        return self.vertices.min()

    @property
    def max(self: PolygonT) -> kgpy.vectors.Cartesian3D:
        return self.vertices.max()

    @property
    @abc.abstractmethod
    def vertices(self: PolygonT) -> kgpy.vectors.Cartesian3D:
        pass

    @property
    def wire(self: PolygonT) -> kgpy.vectors.Cartesian3D:
        vertices = self.vertices
        vertices = np.broadcast_to(vertices, vertices.shape)
        left_vert = np.roll(vertices, -1, axis='vertex')
        right_vert = vertices
        diff = left_vert - right_vert
        t = kgpy.labeled.LinearSpace(0, 1, num=self.num_samples, endpoint=False, axis='wire')
        wire = right_vert + diff * t
        wire = wire.combine_axes(axes=['vertex', 'wire'], axis_new='wire')
        return wire


@dataclasses.dataclass
class RegularPolygon(Polygon):
    radius: kgpy.uncertainty.ArrayLike = 0 * u.mm
    num_sides: kgpy.labeled.ArrayLike = 8
    offset_angle: kgpy.uncertainty.ArrayLike = 0 * u.deg

    @property
    def broadcasted(self: RegularPolygonT):
        out = super().broadcasted
        out = np.broadcast(out, self.radius)
        return out

    @property
    def vertices(self: RegularPolygonT) -> kgpy.vectors.Cartesian3D:
        vertices = kgpy.vectors.Cylindrical(
            radius=self.radius,
            azimuth=kgpy.labeled.LinearSpace(
                start=self.offset_angle,
                stop=360 * u.deg + self.offset_angle,
                num=self.num_sides,
                endpoint=False,
                axis='vertex',
            ),
            z=0 * self.radius,
        ).cartesian
        return self.transform(vertices)

    @property
    def edge_subtent(self: RegularPolygonT) -> kgpy.labeled.ArrayLike:
        """
        Calculate the angle subtended by each edge of the polygon by dividing the angle of a circle (360 degrees) by
        the number of sides in the regular polygon.
        :return: Angle subtended by each edge
        """
        return 360 * u.deg / self.num_sides

    @property
    def half_edge_subtent(self: RegularPolygonT) -> kgpy.labeled.ArrayLike:
        """
        Calculate the angle subtended between a vertex and a point on the center of an edge.
        This is sometimes a more useful quantity than the subtent of an entire edge.
        :return:
        """
        return self.edge_subtent / 2

    @property
    def min_radius(self: RegularPolygonT) -> kgpy.uncertainty.ArrayLike:
        """
        Calculate the distance from the center of the polygon to the center of an edge of a polygon.
        :return: The minimum radius of the polygon.
        """
        return self.radius * np.cos(self.half_edge_subtent)


@dataclasses.dataclass
class IrregularPolygon(Polygon):
    vertices: kgpy.vectors.Cartesian3D = None


@dataclasses.dataclass
class Rectangular(Polygon):
    half_width: kgpy.vectors.Cartesian2D = dataclasses.field(default_factory=lambda: kgpy.vectors.Cartesian2D() * u.mm)

    @property
    def broadcasted(self: RectangularT):
        out = super().broadcasted
        # out = np.broadcast(out, self.half_width_x)
        # out = np.broadcast(out, self.half_width_y)
        return out

    def is_unvignetted(self: RectangularT, position: kgpy.vectors.Cartesian2D) -> kgpy.uncertainty.ArrayLike:
        amin = self.min
        amax = self.max
        m1 = position.x <= amax.x
        m2 = position.x >= amin.x
        m3 = position.y <= amax.y
        m4 = position.y >= amin.y
        is_inside = m1 & m2 & m3 & m4
        if not self.is_obscuration:
            return is_inside
        else:
            return ~is_inside

    # @property
    # def min(self: RectangularT) -> kgpy.vector.Cartesian3D:
    #     result = -self.half_width.to_3d(z=0 * self.half_width.x)
    #     return self.transform.inverse(result)
    #
    # @property
    # def max(self: RectangularT) -> kgpy.vector.Cartesian3D:
    #     result = self.half_width.to_3d(z=0 * self.half_width.x)
    #     return self.transform.inverse(result)

    @property
    def vertices(self: RectangularT) -> kgpy.vectors.Cartesian3D:
        result = kgpy.vectors.Cylindrical(
            radius=np.sqrt(2),
            azimuth=kgpy.labeled.LinearSpace(
                start=0 * u.deg,
                stop=360 * u.deg,
                num=4,
                endpoint=False,
                axis='vertex',
            ) + 45 * u.deg,
            z=0,
        )
        result = result.cartesian * self.half_width.to_3d()
        result = self.transform(result)
        return result


# @dataclasses.dataclass
# class AsymmetricRectangular(Polygon):
#     width_negative: kgpy.vector.Cartesian2D = dataclasses.field(default_factory=lambda: kgpy.vector.Cartesian2D() * u.mm)
#     width_positive: kgpy.vector.Cartesian2D = dataclasses.field(default_factory=lambda: kgpy.vector.Cartesian2D() * u.mm)
#
#     @property
#     def broadcasted(self):
#         out = super().broadcasted
#         out = np.broadcast(out, self.width_x_neg)
#         out = np.broadcast(out, self.width_x_pos)
#         out = np.broadcast(out, self.width_y_neg)
#         out = np.broadcast(out, self.width_y_pos)
#         return out
#
#     @property
#     def vertices(self: AsymmetricRectangularT) -> kgpy.vector.Cartesian3D:
#         result = kgpy.vector.Cartesian3D(
#             x=kgpy.labeled.LinearSpace(0, 1, num=2, axis='vertex_x'),
#             y=kgpy.labeled.LinearSpace(0, 1, num=2, axis='vertex_y'),
#             z=0,
#         )
#
#         result.x = result.x + 0 * result.y
#         result.y = result.y + 0 * result.x
#
#         result = result * (self.width_positive - self.width_negative) + self.width_negative
#
#         result.combine_axes(axes=['vertex_x', 'vertex_y'], axis_new='vertex')
#
#         return self.transform.inverse(result)


@dataclasses.dataclass
class IsoscelesTrapezoid(Polygon):
    apex_offset: kgpy.uncertainty.ArrayLike = 0 * u.mm
    half_width_left: kgpy.uncertainty.ArrayLike = 0 * u.mm
    half_width_right: kgpy.uncertainty.ArrayLike = 0 * u.mm
    wedge_half_angle: kgpy.uncertainty.ArrayLike = 0 * u.deg

    @property
    def vertices(self) -> kgpy.vectors.Cartesian3D:
        m = np.tan(self.wedge_half_angle)
        zero = 0 * self.apex_offset + 0 * self.half_width_left + 0 * self.half_width_right + 0 * m
        left_x, left_y = -self.half_width_left + zero, -m * (self.apex_offset + self.half_width_left) + zero
        right_x, right_y = self.half_width_right + zero, -m * (self.apex_offset - self.half_width_right) + zero
        vertices = kgpy.vectors.Cartesian3D(
            x=np.stack([left_x, right_x, right_x, left_x], axis='vertex'),
            y=np.stack([left_y, right_y, -right_y, -left_y], axis='vertex'),
            z=zero,
        )
        return self.transform(vertices)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.apex_offset)
        out = np.broadcast(out, self.half_width_left)
        out = np.broadcast(out, self.half_width_right)
        out = np.broadcast(out, self.wedge_half_angle)
        return out

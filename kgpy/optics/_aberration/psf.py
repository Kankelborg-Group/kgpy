import typing as typ
import abc
import dataclasses
import numpy as np
import scipy.interpolate
import astropy.units as u
import kgpy
import kgpy.mixin
import kgpy.vectors
import kgpy.function

__all__ = []


@dataclasses.dataclass
class PSF(abc.ABC):
    pass


# class Axis(kgpy.mixin.AutoAxis):
#
#     def __init__(self):
#         super().__init__()
#         # self.velocity_los = self.auto_axis_index()
#         self.wavelength = self.auto_axis_index()
#         self.position_y = self.auto_axis_index()
#         self.position_x = self.auto_axis_index()
#         self.field_y = self.auto_axis_index()
#         self.field_x = self.auto_axis_index()
#
#     @property
#     def position_xy(self) -> typ.Tuple[int, int]:
#         return self.position_x, self.position_y
#
#     @property
#     def field_xy(self) -> typ.Tuple[int, int]:
#         return self.field_x, self.field_y
#
#     @property
#     def latex_names(self) -> typ.List[str]:
#         names = [None] * self.ndim
#         names[self.field_x] = 'field $x$'
#         names[self.field_y] = 'field $y$'
#         names[self.position_x] = 'position $x$'
#         names[self.position_y] = 'position $y$'
#         names[self.wavelength] = 'wavelength'
#         # names[self.velocity_los] = 'LOS velocity'
#         return names
#
#
# @dataclasses.dataclass
# class Grid(kgpy.mixin.Copyable):
#     axis: typ.ClassVar[Axis] = Axis()
#     field: kgpy.grid.RegularGrid2D = dataclasses.field(default_factory=kgpy.grid.RegularGrid2D)
#     position: kgpy.grid.Grid2D = dataclasses.field(default_factory=kgpy.grid.Grid2D)
#     wavelength: kgpy.grid.Grid1D = dataclasses.field(
#         default_factory=lambda: kgpy.grid.RegularGrid1D(min=0 * u.nm, max=0 * u.nm)
#     )
#     # velocity_los: kgpy.grid.Grid1D = dataclasses.field(
#     #     default_factory=lambda: kgpy.grid.RegularGrid1D(min=0 * u.km / u.s, max=0 * u.km / u.s)
#     # )
#
#     @property
#     def shape(self) -> typ.Tuple[int, ...]:
#         return np.broadcast(
#             np.expand_dims(self.field.points.x, self.axis.perp_axes(self.axis.field_x)),
#             np.expand_dims(self.field.points.y, self.axis.perp_axes(self.axis.field_y)),
#             np.expand_dims(self.position.points.x, self.axis.perp_axes(self.axis.position_x)),
#             np.expand_dims(self.position.points.y, self.axis.perp_axes(self.axis.position_y)),
#             np.expand_dims(self.wavelength.points, self.axis.perp_axes(self.axis.wavelength)),
#             # np.expand_dims(self.velocity_los.points, self.axis.perp_axes(self.axis.velocity_los)),
#         ).shape
#
#     @property
#     def points_field(self) -> kgpy.vector.Vector2D:
#         return self.field.mesh(shape=self.shape, new_axes=self.axis.perp_axes([self.axis.field_x, self.axis.field_y]))
#
#     @property
#     def points_position(self) -> kgpy.vector.Vector2D:
#         return self.position.mesh(shape=self.shape, new_axes=self.axis.perp_axes([self.axis.position_x, self.axis.position_y]))
#
#     @property
#     def points_wavelength(self) -> u.Quantity:
#         return self.wavelength.mesh(shape=self.shape, new_axes=self.axis.perp_axes(self.axis.wavelength))
#
#     # @property
#     # def points_velocity_los(self) -> u.Quantity:
#     #     return self.velocity_los.mesh(shape=self.shape, new_axes=self.axis.perp_axes(self.axis.velocity_los))
#
#     def view(self) -> 'Grid':
#         other = super().view()  # type: Grid
#         other.field = self.field
#         other.position = self.position
#         other.wavelength = self.wavelength
#         # other.velocity_los = self.velocity_los
#         return other
#
#     def copy(self) -> 'Grid':
#         other = super().copy()      # type: Grid
#         other.field = self.field.copy()
#         other.position = self.position.copy()
#         other.wavelength = self.wavelength.copy()
#         # other.velocity_los = self.velocity_los.copy()
#         return other


@dataclasses.dataclass
class DiscretePSF(PSF):

    data: kgpy.function.Array

    @classmethod
    def from_pupil_function(cls, pupil_function: kgpy.function.Array):

        psf = np.fft.fft2(pupil_function.data, axes=grid.axis.position_xy)
        psf = np.fft.fftshift(psf, axes=grid.axis.position_xy)
        psf = psf * np.conjugate(psf)
        psf = psf.real
        psf = psf / psf.sum(axis=grid.axis.position_xy, keepdims=True)

        frequency = kgpy.vector.Vector2D(
            x=np.fft.fftfreq(n=psf.shape[cls.axis.position_x], d=grid.position.step_size.x),
            y=np.fft.fftfreq(n=psf.shape[cls.axis.position_y], d=grid.position.step_size.y),
        )
        # frequency.x = np.expand_dims(frequency.x, axis=cls.axis.perp_axes(cls.axis.position_x))
        # frequency.y = np.expand_dims(frequency.y, axis=cls.axis.perp_axes(cls.axis.position_y))

        position = np.arcsin(frequency * grid.wavelength.points).to(u.arcsec)

        return cls(
            data=psf,
            grid=Grid(
                field=grid.field.copy(),
                position=kgpy.grid.RectilinearGrid2D(points=position),
                wavelength=grid.wavelength.copy(),
                # velocity_los=grid.velocity_los.copy(),
            )
        )

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.grid.shape[:~(self.axis.ndim - 1)]

    @property
    def _interpolator(self) -> np.ndarray:

        data = np.broadcast_to(self.data, shape=self.grid.shape)

        shape = self.shape
        if not shape:
            shape = 1,
            data = data[np.newaxis]

        interpolator = np.empty(shape=shape, dtype=scipy.interpolate.RegularGridInterpolator)

        points_field_x = np.broadcast_to(self.grid.field.points.x, shape=shape + (self.grid.field.num_samples_normalized.x,))
        points_field_y = np.broadcast_to(self.grid.field.points.y, shape=shape + (self.grid.field.num_samples_normalized.y,))
        points_position_x = np.broadcast_to(self.grid.position.points.x, shape=shape + (self.grid.position.num_samples_normalized.x,))
        points_position_y = np.broadcast_to(self.grid.position.points.y, shape=shape + (self.grid.position.num_samples_normalized.y,))
        points_wavelength = np.broadcast_to(self.grid.wavelength.points, shape=shape + (self.grid.wavelength.num_samples,))
        # points_velocity_los = np.broadcast_to(self.grid.velocity_los.points, shape=shape + (self.grid.velocity_los.num_samples,))

        for i in range(interpolator.size):
            index = np.unravel_index(i, shape=shape)

            points = [None] * self.axis.ndim
            points[self.axis.field_x] = points_field_x[index]
            points[self.axis.field_y] = points_field_y[index]
            points[self.axis.position_x] = points_position_x[index]
            points[self.axis.position_y] = points_position_y[index]
            points[self.axis.wavelength] = points_wavelength[index]
            # points[self.axis.velocity_los] = points_velocity_los[index].value
            # for axis in self.axis.all:
            #     if len(points[axis]) < 2:
            #         del points[axis]

            interpolator[index] = scipy.interpolate.RegularGridInterpolator(
                points=points,
                values=data[index].value,
            )

        return interpolator

    def __call__(
            self,
            field: kgpy.vector.Vector2D,
            position: kgpy.vector.Vector2D,
            wavelength: u.Quantity,
    ):
        if not self.shape:
            field = field[np.newaxis]
            position = position[np.newaxis]
            wavelength = wavelength[np.newaxis]

        shape = np.broadcast(field.x, field.y, position.x, position.y, wavelength).shape

        xi = [None] * self.axis.ndim
        xi[self.axis.field_x] = field.x.to(self.grid.field.points.x.unit).value
        xi[self.axis.field_y] = field.y.to(self.grid.field.points.y.unit).value
        xi[self.axis.position_x] = position.x.to(self.grid.position.points.x.unit).value
        xi[self.axis.position_y] = position.y.to(self.grid.position.points.y.unit).value
        xi[self.axis.wavelength] = wavelength.to(self.grid.wavelength.points.unit).value
        xi = np.stack(xi, axis=~0)

        result = np.empty(shape)

        interpolator = self._interpolator

        for i in range(interpolator.size):
            index = np.unravel_index(i, interpolator.shape)

            result[index] = interpolator[index](xi[index])

        return result << self.data.unit

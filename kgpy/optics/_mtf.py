import typing as typ
import functools
import dataclasses
import numpy as np
import scipy.interpolate
import astropy.units as u
import kgpy.mixin
import kgpy.vectors

__all__ = ['MTF']


class Axis(kgpy.mixin.AutoAxis):
    ndim_pupil: typ.ClassVar[int] = 2
    ndim_field: typ.ClassVar[int] = 2

    def __init__(self):
        super().__init__()
        self.velocity_los = self.auto_axis_index()
        self.wavelength = self.auto_axis_index()
        self.frequency_y = self.auto_axis_index()
        self.frequency_x = self.auto_axis_index()
        self.field_y = self.auto_axis_index()
        self.field_x = self.auto_axis_index()
        # self.wavelength = self.auto_axis_index()

    @property
    def frequency_xy(self) -> typ.Tuple[int, int]:
        return self.frequency_x, self.frequency_y

    @property
    def field_xy(self) -> typ.Tuple[int, int]:
        return self.field_x, self.field_y

    @property
    def latex_names(self) -> typ.List[str]:
        names = [None] * self.ndim
        names[self.field_x] = 'field $x$'
        names[self.field_y] = 'field $y$'
        names[self.frequency_x] = 'frequency $x$'
        names[self.frequency_y] = 'frequency $y$'
        names[self.wavelength] = 'wavelength'
        names[self.velocity_los] = 'LOS velocity'
        return names


@dataclasses.dataclass
class MTF:

    axis: typ.ClassVar[Axis] = Axis()

    unit_field: typ.ClassVar[u.Unit] = u.arcsec
    unit_frequency: typ.ClassVar[u.Unit] = 1 / u.arcsec
    unit_wavelength: typ.ClassVar[u.Unit] = u.AA
    unit_velocity_los: typ.ClassVar[u.Unit] = u.km / u.s

    data_field: kgpy.vectors.Cartesian2D
    data_frequency: kgpy.vectors.Cartesian2D
    data_wavelength: u.Quantity
    data_velocity_los: u.Quantity
    data_mtf: u.Quantity



    @property
    def grid_shape(self) -> typ.Tuple[int, ...]:
        return np.broadcast(
            self.data_field,
            self.data_frequency,
            self.data_wavelength,
            self.data_velocity_los,
            self.data_mtf,
        ).shape

    @property
    def data_final_field(self) -> kgpy.vectors.Cartesian2D:
        return np.broadcast_to(self.data_field, shape=self.grid_shape, subok=True)

    @property
    def data_final_frequency(self) -> kgpy.vectors.Cartesian2D:
        return np.broadcast_to(self.data_frequency, shape=self.grid_shape, subok=True)

    @property
    def data_final_wavelength(self) -> u.Quantity:
        return np.broadcast_to(self.data_wavelength, shape=self.grid_shape, subok=True)

    @property
    def data_final_velocity_los(self) -> u.Quantity:
        return np.broadcast_to(self.data_velocity_los, shape=self.grid_shape, subok=True)

    @property
    def data_final_mtf(self) -> u.Quantity:
        return np.broadcast_to(self.data_mtf, shape=self.grid_shape, subok=True)

    @property
    def interpolator(self) -> scipy.interpolate.LinearNDInterpolator:

        mtf = self.data_final_mtf
        mask = np.broadcast_to(mtf.sum(self.axis.frequency_xy, keepdims=True) > 0, self.grid_shape)

        data_field = self.data_final_field[mask].to(self.unit_field)
        data_frequency = self.data_final_frequency[mask].to(self.unit_frequency)

        points = [None] * self.axis.ndim
        points[self.axis.field_x] = data_field.x.value
        points[self.axis.field_y] = data_field.y.value
        points[self.axis.frequency_x] = data_frequency.x.value
        points[self.axis.frequency_y] = data_frequency.y.value
        points[self.axis.wavelength] = self.data_final_wavelength[mask].to(self.unit_wavelength).value
        points[self.axis.velocity_los] = self.data_final_velocity_los[mask].to(self.unit_velocity_los).value
        del points[self.axis.velocity_los]
        points = np.stack(points, axis=~0)

        return scipy.interpolate.LinearNDInterpolator(
            points=points,
            values=mtf[mask].value,
            rescale=True,
        )

    def __call__(
            self,
            field: kgpy.vectors.Cartesian2D,
            frequency: kgpy.vectors.Cartesian2D,
            wavelength: u.Quantity,
            velocity_los: u.Quantity = 0 * u.km / u.s,
    ):
        field = field.to(self.unit_field)
        frequency = frequency.to(self.unit_frequency)
        wavelength = wavelength.to(self.unit_wavelength)
        velocity_los = velocity_los.to(self.unit_velocity_los)

        points = [None] * self.axis.ndim
        points[self.axis.field_x] = field.x.value
        points[self.axis.field_y] = field.y.value
        points[self.axis.frequency_x] = frequency.x.value
        points[self.axis.frequency_y] = frequency.y.value
        points[self.axis.wavelength] = wavelength.value
        points[self.axis.velocity_los] = velocity_los.value
        del points[self.axis.velocity_los]

        return self.interpolator(*points) * u.dimensionless_unscaled


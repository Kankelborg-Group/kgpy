import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import vg
from . import surface

__all__ = ['Rays']


@dataclasses.dataclass
class Axis:

    def __post_init__(self):
        self.num_axes = 0

        self.components = self.auto_axis_index()
        self.config = self.auto_axis_index()
        self.pupil = self.auto_axis_index()
        self.field = self.auto_axis_index()
        self.wavlen = self.auto_axis_index()

    def auto_axis_index(self):
        i = ~self.num_axes
        self.num_axes += 1
        return i


class Components:
    x = ..., 0
    y = ..., 1
    z = ..., 2


@dataclasses.dataclass
class Rays:

    axis = Axis()
    components = Components()

    wavelength: u.Quantity
    position: u.Quantity
    direction: u.Quantity
    polarization: u.Quantity
    surface_normal: u.Quantity
    vignetted_mask: np.ndarray
    error_mask: np.ndarray
    index_of_refraction: u.Quantity

    @classmethod
    def from_field_angles(
            cls,
            wavelength: u.Quantity,
            field_x: u.Quantity,
            field_y: u.Quantity,
            pupil_x: u.Quantity,
            pupil_y: u.Quantity,
    ) -> 'Rays':
        sh = np.broadcast(wavelength, pupil_x, pupil_y, field_x, field_y, ).shape

        self = cls.zeros(sh)

        self.position[cls.x] = pupil_x
        self.position[cls.y] = pupil_y

        self.direction[cls.x] = np.cos(field_x)
        self.direction[cls.y] = np.cos(field_y)
        self.direction[cls.z] = np.sqrt(1 - np.square(self.direction[cls.x]) - np.square(self.direction[cls.y]))

        return self

    @classmethod
    def zeros(cls, shape: typ.Tuple[int, ...]):
        vsh = shape + (3, )
        ssh = shape + (1, )

        return cls(
            wavelength=np.empty(ssh) << u.nm,
            position=np.empty(vsh) << u.mm,
            direction=np.empty(vsh) << u.dimensionless_unscaled,
            surface_normal=np.empty(vsh) << u.dimensionless_unscaled,
            vignetted_mask=np.empty(ssh, dtype=np.bool),
            error_mask=np.empty(ssh, dtype=np.bool),

        )

    def tilt_decenter(self, transform: surface.coordinate.TiltDecenter) -> 'Rays':
        return type(self)(
            wavelength=self.wavelength.copy(),
            position=transform.apply(self.position),
            direction=transform.apply(self.direction, decenter=False),
            surface_normal=transform.apply(self.surface_normal, decenter=False),
            vignetted_mask=self.vignetted_mask.copy(),
            error_mask=self.error_mask.copy(),
            polarization=self.polarization.copy(),
            index_of_refraction=self.index_of_refraction.copy(),
        )

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return np.broadcast(
            self.wavelength,
            self.position[0],
            self.direction[0],
            self.surface_normal[0],
            self.vignetted_mask,
            self.error_mask,
            self.polarization,
            self.index_of_refraction,
        ).shape

    @property
    def px(self) -> u.Quantity:
        return self.position[self.components.x]

    @px.setter
    def px(self, value: u.Quantity):
        self.position[self.components.x] = value

    @property
    def py(self) -> u.Quantity:
        return self.position[self.components.y]

    @py.setter
    def py(self, value: u.Quantity):
        self.position[self.components.y] = value

    @property
    def pz(self) -> u.Quantity:
        return self.position[self.components.z]

    @pz.setter
    def pz(self, value: u.Quantity):
        self.position[self.components.z] = value

    def copy(self) -> 'Rays':
        return Rays(
            wavelength=self.wavelength.copy(),
            position=self.position.copy(),
            direction=self.direction.copy(),
            surface_normal=self.surface_normal.copy(),
            vignetted_mask=self.vignetted_mask.copy(),
            error_mask=self.error_mask.copy(),
            polarization=self.polarization.copy(),
            index_of_refraction=self.index_of_refraction.copy(),
        )

    @classmethod
    def empty(cls, sh: typ.Tuple[int, int, int, int, int, int, int]) -> 'Rays':
        vsh = sh + (3,)
        ssh = sh + (1,)

        return cls(
            input_coordinates=(np.empty(sh[cls.axis.field_x]) * u.arcmin,
                               np.empty(sh[cls.axis.field_y]) * u.arcmin,
                               np.empty(sh[cls.axis.wavl]) * u.AA),
            position=np.empty(vsh) * u.mm,
            direction=np.empty(vsh),
            surface_normal=np.empty(vsh),
            opd=np.empty(ssh) * u.mm,
            mask=np.empty(ssh, dtype=np.bool),
        )

    @property
    def input_grid(self) -> typ.Tuple[u.Quantity, u.Quantity, u.Quantity]:
        x, y, w = self.input_coordinates
        xx, yy, ww = np.meshgrid(x.value, y.value, w.value, indexing='ij')
        xx, yy, ww = xx << x.unit, yy << y.unit, ww << w.unit
        return xx, yy, ww

    @property
    def input_grid_raveled(self) -> typ.Tuple[u.Quantity, u.Quantity, u.Quantity]:
        x, y, w = self.input_grid
        return x.ravel(), y.ravel(), w.ravel()

    def unvignetted_mean(self, axis=None) -> 'Rays':
        uvm = self.unvignetted_mask

        norm = np.sum(uvm, axis=axis, keepdims=True)

        return type(self)(
            input_coordinates=self.input_coordinates,
            position=np.nansum(self.position.value * uvm, axis=axis, keepdims=True) * self.position.unit / norm,
            direction=np.nansum(self.direction * uvm, axis=axis, keepdims=True) / norm,
            surface_normal=np.nansum(self.surface_normal * uvm, axis=axis, keepdims=True) / norm,
            opd=np.nansum(self.opd.value * uvm, axis=axis, keepdims=True) * self.opd.unit / norm,
            mask=np.max(self.mask, axis=axis, keepdims=True),
        )

    @property
    def unvignetted_mask(self):
        sl = [slice(None)] * len(self.mask.shape)
        sl[self.axis.surf] = slice(~0, None)
        unvignetted_mask = self.mask[tuple(sl)]
        return unvignetted_mask

    @property
    def pupil_mean(self):
        return self.unvignetted_mean(axis=(self.axis.pupil_x, self.axis.pupil_y))

    @property
    def field_mean(self):
        return self.unvignetted_mean(axis=(self.axis.field_x, self.axis.field_y))

    @property
    def input_angle(self):
        return self._angle_between_ray_and_surface(-1)

    @property
    def output_angle(self):
        return self._angle_between_ray_and_surface(0)

    def _angle_between_ray_and_surface(self, offset: int):
        sl = [slice(None)] * len(self.direction.shape)
        sl[self.axis.surf] = slice(1, None)

        normal = np.roll(self.surface_normal, offset, axis=self.axis.surf)
        direction = self.direction

        sh = normal.shape

        normal = -np.reshape(normal, (-1, 3))
        direction = np.reshape(direction, (-1, 3))

        x = vg.angle(normal, direction) * u.deg

        x = np.reshape(x, sh[:~0] + (1,))

        return x

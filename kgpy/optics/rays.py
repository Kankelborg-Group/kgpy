import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from . import coordinate

__all__ = ['Rays']


@dataclasses.dataclass
class Axis:

    def __post_init__(self):
        self.num_axes = 0

        self.components = self.auto_axis_index()
        self.config = self.auto_axis_index()

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
    index_of_refraction: u.Quantity
    vignetted_mask: np.ndarray
    error_mask: np.ndarray

    @classmethod
    def zeros(cls, shape: typ.Tuple[int, ...] = ()):
        vsh = shape + (3, )
        ssh = shape + (1, )

        direction = np.zeros(vsh) << u.dimensionless_unscaled
        polarization = np.zeros(vsh) << u.dimensionless_unscaled
        normal = np.zeros(vsh) << u.dimensionless_unscaled

        direction[cls.components.z] = 1
        polarization[cls.components.x] = 1
        normal[cls.components.z] = 1

        return cls(
            wavelength=np.zeros(ssh) << u.nm,
            position=np.zeros(vsh) << u.mm,
            direction=direction,
            polarization=polarization,
            surface_normal=normal,
            index_of_refraction=np.ones(ssh) << u.dimensionless_unscaled,
            vignetted_mask=np.zeros(ssh, dtype=np.bool),
            error_mask=np.zeros(ssh, dtype=np.bool),
        )

    def tilt_decenter(self, transform: coordinate.TiltDecenter) -> 'Rays':
        return type(self)(
            wavelength=self.wavelength.copy(),
            position=transform(self.position, num_extra_dims=1),
            direction=transform(self.direction, decenter=False, num_extra_dims=1),
            polarization=self.polarization.copy(),
            surface_normal=transform(self.surface_normal, decenter=False, num_extra_dims=1),
            index_of_refraction=self.index_of_refraction.copy(),
            vignetted_mask=self.vignetted_mask.copy(),
            error_mask=self.error_mask.copy(),
        )

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return np.broadcast(
            self.wavelength,
            self.position,
            self.direction,
            self.surface_normal,
            self.vignetted_mask,
            self.error_mask,
            self.polarization,
            self.index_of_refraction,
        ).shape[:~0]

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
import abc
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import mixin, vector, optimization
from ..rays import Rays

__all__ = [
    'Sag',
    'Standard',
    'Toroidal',
    'SlopeErrorRMS',
    'RoughnessRMS',
    'RippleRMS',
]


@dataclasses.dataclass
class SlopeErrorRMS(
    mixin.Copyable,
):
    value: u.Quantity = 0 * u.urad
    length_integration: u.Quantity = 0 * u.mm
    length_sample: u.Quantity = 0 * u.mm

    def view(self) -> 'SlopeErrorRMS':
        other = super().view()  # type: SlopeErrorRMS
        other.value = self.value
        other.length_integration = self.length_integration
        other.length_sample = self.length_sample
        return other

    def copy(self) -> 'SlopeErrorRMS':
        other = super().copy()  # type: SlopeErrorRMS
        other.value = self.value.copy()
        other.length_integration = self.length_integration.copy()
        other.length_sample = self.length_sample.copy()
        return other


@dataclasses.dataclass
class RoughnessRMS(
    mixin.Copyable,
):
    value: u.Quantity = 0 * u.mm
    periods_min: u.Quantity = 0 * u.mm
    periods_max: u.Quantity = 0 * u.mm

    def view(self) -> 'RoughnessRMS':
        other = super().view()  # type: RoughnessRMS
        other.value = self.value
        other.periods_min = self.periods_min
        other.periods_max = self.periods_max
        return other

    def copy(self) -> 'RoughnessRMS':
        other = super().copy()  # type: RoughnessRMS
        other.value = self.value.copy()
        other.periods_min = self.periods_min.copy()
        other.periods_max = self.periods_max.copy()
        return other


class RippleRMS(RoughnessRMS):
    pass


@dataclasses.dataclass
class Sag(
    mixin.Broadcastable,
    mixin.Copyable,
    abc.ABC,
):

    @abc.abstractmethod
    def __call__(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0, ) -> u.Quantity:
        pass

    @abc.abstractmethod
    def normal(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0, ) -> vector.Vector3D:
        pass


@dataclasses.dataclass
class Standard(Sag):
    radius: u.Quantity = np.inf * u.mm
    conic: u.Quantity = 0 * u.dimensionless_unscaled

    @property
    def curvature(self) -> u.Quantity:
        return np.where(np.isinf(self.radius), 0 / u.mm, 1 / self.radius)

    def __call__(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0, ) -> u.Quantity:
        extra_dims_slice = (Ellipsis,) + num_extra_dims * (np.newaxis,)
        r2 = np.square(x) + np.square(y)
        c = self.curvature[extra_dims_slice]
        conic = self.conic[extra_dims_slice]
        sz = c * r2 / (1 + np.sqrt(1 - (1 + conic) * np.square(c) * r2))
        mask = r2 >= np.square(self.radius[extra_dims_slice])
        sz[mask] = 0
        return sz

    def __eq__(self, other: 'Standard'):
        if not super().__eq__(other):
            return False
        if (self.radius != other.radius).any():
            return False
        if (self.conic != other.conic).any():
            return False
        return True

    def normal(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0, ) -> vector.Vector3D:
        extra_dims_slice = (Ellipsis,) + num_extra_dims * (np.newaxis,)
        x2, y2 = np.square(x), np.square(y)
        c = self.curvature[extra_dims_slice]
        c2 = np.square(c)
        conic = self.conic[extra_dims_slice]
        g = np.sqrt(1 - (1 + conic) * c2 * (x2 + y2))
        dzdx, dzdy = c * x / g, c * y / g
        mask = (x2 + y2) >= np.square(self.radius[extra_dims_slice])
        dzdx[mask] = 0
        dzdy[mask] = 0
        return vector.Vector3D(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled
        ).normalize()

    def view(self) -> 'Standard':
        other = super().view()  # type: Standard
        other.radius = self.radius
        other.conic = self.conic
        return other

    def copy(self) -> 'Standard':
        other = super().copy()  # type: Standard
        other.radius = self.radius.copy()
        other.conic = self.conic.copy()
        return other

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.radius)
        out = np.broadcast(out, self.conic)
        return out


@dataclasses.dataclass
class Toroidal(Standard):
    radius_of_rotation: u.Quantity = 0 * u.mm

    def __call__(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0, ) -> u.Quantity:
        extra_dims_slice = (Ellipsis,) + num_extra_dims * (np.newaxis,)
        x2 = np.square(x)
        y2 = np.square(y)
        c = self.curvature[extra_dims_slice]
        r = self.radius_of_rotation[extra_dims_slice]
        mask = np.abs(x) > r
        conic = self.conic[extra_dims_slice]
        zy = c * y2 / (1 + np.sqrt(1 - (1 + conic) * np.square(c) * y2))
        z = r - np.sqrt(np.square(r - zy) - x2)
        z[mask] = (r - np.sqrt(np.square(r - zy) - np.square(r)))[mask]
        return z

    def __eq__(self, other: 'Toroidal') -> bool:
        if not super().__eq__(other):
            return False
        if (self.radius_of_rotation != other.radius_of_rotation).any():
            return False
        return True

    def normal(self, x: u.Quantity, y: u.Quantity, num_extra_dims: int = 0, ) -> vector.Vector3D:
        extra_dims_slice = (Ellipsis,) + num_extra_dims * (np.newaxis,)
        x2 = np.square(x)
        y2 = np.square(y)
        c = self.curvature[extra_dims_slice]
        c2 = np.square(c)
        r = self.radius_of_rotation[extra_dims_slice]
        conic = self.conic[extra_dims_slice]
        g = np.sqrt(1 - (1 + conic) * c2 * y2)
        zy = c * y2 / (1 + g)
        f = np.sqrt(np.square(r - zy) - x2)
        dzdx = x / f
        dzydy = c * y / g
        dzdy = (r - zy) * dzydy / f
        mask = np.abs(x) > r
        dzdx[mask] = 0
        dzdy[mask] = 0
        return vector.Vector3D(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled
        ).normalize()

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.radius_of_rotation)
        return out

    def view(self) -> 'Toroidal':
        other = super().view()  # type: Toroidal
        other.radius_of_rotation = self.radius_of_rotation
        return other

    def copy(self) -> 'Toroidal':
        other = super().copy()  # type: Toroidal
        other.radius_of_rotation = self.radius_of_rotation.copy()
        return other

import typing as typ
import abc
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.mixin
import kgpy.uncertainty
import kgpy.vectors

__all__ = [
    'Sag',
    'Standard',
    'Toroidal',
    'SlopeErrorRMS',
    'RoughnessRMS',
    'RippleRMS',
]

SagT = typ.TypeVar('SagT', bound='Sag')
StandardT = typ.TypeVar('StandardT', bound='Standard')
ToroidalT = typ.TypeVar('ToroidalT', bound='Toroidal')


@dataclasses.dataclass
class SlopeErrorRMS(
    kgpy.mixin.Copyable,
):
    value: u.Quantity = 0 * u.urad
    length_integration: u.Quantity = 0 * u.mm
    length_sample: u.Quantity = 0 * u.mm


@dataclasses.dataclass
class RoughnessRMS(
    kgpy.mixin.Copyable,
):
    value: u.Quantity = 0 * u.mm
    periods_min: u.Quantity = 0 * u.mm
    periods_max: u.Quantity = 0 * u.mm


class RippleRMS(RoughnessRMS):
    pass


@dataclasses.dataclass
class Sag(
    kgpy.mixin.Broadcastable,
    kgpy.mixin.Copyable,
    abc.ABC,
):

    @abc.abstractmethod
    def __call__(self: SagT, position: kgpy.vectors.Cartesian2D) -> kgpy.uncertainty.ArrayLike:
        pass

    @abc.abstractmethod
    def normal(self: SagT, position: kgpy.vectors.Cartesian2D) -> kgpy.vectors.Cartesian3D:
        pass


@dataclasses.dataclass
class Standard(Sag):
    radius: kgpy.uncertainty.ArrayLike = np.inf * u.mm
    conic: kgpy.uncertainty.ArrayLike = 0 * u.dimensionless_unscaled

    @property
    def curvature(self) -> kgpy.uncertainty.ArrayLike:
        return 1 / self.radius

    def __call__(self, position: kgpy.vectors.Cartesian2D) -> kgpy.uncertainty.ArrayLike:
        r2 = np.square(position.x) + np.square(position.y)
        c = self.curvature
        sz = c * r2 / (1 + np.sqrt(1 - (1 + self.conic) * np.square(c) * r2))
        mask = r2 >= np.square(self.radius)
        sz[mask] = 0
        return sz

    def __eq__(self, other: 'Standard'):
        if not isinstance(other, type(self)):
            return False
        if not super().__eq__(other):
            return False
        if not np.all(self.radius == other.radius):
            return False
        if not np.all(self.conic == other.conic):
            return False
        return True

    def normal(self, position: kgpy.vectors.Cartesian2D) -> kgpy.vectors.Cartesian3D:
        x2, y2 = np.square(position.x), np.square(position.y)
        c = self.curvature
        c2 = np.square(c)
        g = np.sqrt(1 - (1 + self.conic) * c2 * (x2 + y2))
        dzdx, dzdy = c * position.x / g, c * position.y / g
        mask = (x2 + y2) >= np.square(self.radius)
        dzdx[mask] = 0
        dzdy[mask] = 0
        return kgpy.vectors.Cartesian3D(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled
        ).normalized

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.radius)
        out = np.broadcast(out, self.conic)
        return out


@dataclasses.dataclass
class Toroidal(Standard):
    radius_of_rotation: kgpy.uncertainty.ArrayLike = 0 * u.mm

    def __call__(self: ToroidalT, position: kgpy.vectors.Cartesian2D) -> kgpy.uncertainty.ArrayLike:
        x2 = np.square(position.x)
        y2 = np.square(position.y)
        c = self.curvature
        r = self.radius_of_rotation
        mask = np.abs(position.x) > r
        conic = self.conic
        zy = c * y2 / (1 + np.sqrt(1 - (1 + conic) * np.square(c) * y2))
        z = r - np.sqrt(np.square(r - zy) - x2)
        z[mask] = (r - np.sqrt(np.square(r - zy) - np.square(r)))[mask]
        return z

    def __eq__(self: ToroidalT, other: ToroidalT) -> bool:
        if not super().__eq__(other):
            return False
        if not np.all(self.radius_of_rotation == other.radius_of_rotation):
            return False
        return True

    def normal(self: ToroidalT, position: kgpy.vectors.Cartesian2D) -> kgpy.vectors.Cartesian3D:
        x2 = np.square(position.x)
        y2 = np.square(position.y)
        c = self.curvature
        c2 = np.square(c)
        r = self.radius_of_rotation
        conic = self.conic
        g = np.sqrt(1 - (1 + conic) * c2 * y2)
        zy = c * y2 / (1 + g)
        f = np.sqrt(np.square(r - zy) - x2)
        dzdx = position.x / f
        dzydy = c * position.y / g
        dzdy = (r - zy) * dzydy / f
        mask = np.abs(position.x) > r
        dzdx[mask] = 0
        dzdy[mask] = 0
        return kgpy.vectors.Cartesian3D(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled
        ).normalized

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.radius_of_rotation)
        return out

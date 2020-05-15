import abc
import dataclasses
import typing as typ
import astropy.units as u
from ... import mixin, Rays

__all__ = ['Material', 'NoMaterial']


@dataclasses.dataclass
class Material(mixin.ZemaxCompatible, mixin.Broadcastable, abc.ABC):

    @abc.abstractmethod
    def index_of_refraction(self, wavelength: u.Quantity, polarization: typ.Optional[u.Quantity]) -> u.Quantity:
        pass

    @property
    @abc.abstractmethod
    def propagation_signum(self) -> float:
        pass


@dataclasses.dataclass
class NoMaterial(Material):

    def to_zemax(self) -> 'NoMaterial':
        from kgpy.optics import zemax
        return zemax.system.surface.material.NoMaterial()

    def index_of_refraction(self, wavelength: u.Quantity, polarization: typ.Optional[u.Quantity]):
        return 1 * u.dimensionless_unscaled

    @property
    def propagation_signum(self) -> float:
        return 1.

import abc
import dataclasses
import typing as typ
import astropy.units as u
import kgpy.mixin
from .. import ZemaxCompatible

__all__ = ['Material']


@dataclasses.dataclass
class Material(ZemaxCompatible, kgpy.mixin.Broadcastable, abc.ABC):

    @abc.abstractmethod
    def index_of_refraction(self, wavelength: u.Quantity, polarization: typ.Optional[u.Quantity]) -> u.Quantity:
        pass

    @property
    @abc.abstractmethod
    def propagation_signum(self) -> float:
        pass

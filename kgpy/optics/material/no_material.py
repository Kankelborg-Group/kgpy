import dataclasses
import typing as typ
from astropy import units as u
from . import Material

__all__ = ['NoMaterial']


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

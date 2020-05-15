import dataclasses
from astropy import units as u
from . import Material


@dataclasses.dataclass
class Mirror(Material):

    thickness: u.Quantity = 0 * u.mm

    def to_zemax(self) -> 'Mirror':
        from kgpy.optics import zemax
        return zemax.system.surface.material.Mirror(thickness=self.thickness)

    def index_of_refraction(self, wavelength: u.Quantity, polarization: typ.Optional[u.Quantity]) -> u.Quantity:
        return 1 * u.dimensionless_unscaled

    @property
    def propagation_signum(self) -> float:
        return -1.

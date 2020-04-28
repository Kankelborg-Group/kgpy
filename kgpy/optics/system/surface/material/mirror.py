import dataclasses
from astropy import units as u
from .material import Material


@dataclasses.dataclass
class Mirror(Material):

    thickness: u.Quantity = 0 * u.mm

    def to_zemax(self) -> 'Mirror':
        from kgpy.optics import zemax
        return zemax.system.surface.material.Mirror(thickness=self.thickness)

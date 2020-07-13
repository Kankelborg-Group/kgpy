import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import kgpy.vector
from kgpy.vector import x, y, z
from .. import Rays, material, aperture
from . import Standard

__all__ = ['DiffractionGrating']

MaterialT = typ.TypeVar('MaterialT', bound=material.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture.Aperture)


@dataclasses.dataclass
class DiffractionGrating(Standard[MaterialT, ApertureT]):

    diffraction_order: int = 1
    groove_density: u.Quantity = 0 * (1 / u.mm)

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'diffraction_order': self.diffraction_order,
            'groove_density': self.groove_density,
        })
        return args

    def to_zemax(self) -> 'Standard':
        from kgpy.optics import zemax
        return zemax.system.surface.Standard(**self.__init__args)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.diffraction_order,
            self.groove_density,
        )

    def groove_normal(self, sx: u.Quantity, sy: u.Quantity) -> u.Quantity:
        return u.Quantity([0 << 1 / u.mm, self.groove_density, 0 << 1 / u.mm])

    def propagate_rays(self, rays: Rays, is_first_surface: bool = False, is_final_surface: bool = False, ) -> Rays:

        if not is_first_surface:
            rays = rays.tilt_decenter(~self.transform_before)
            rays.position = self.calc_intercept(rays)

            n1 = rays.index_of_refraction
            n2 = self.material.index_of_refraction(rays.wavelength, rays.polarization)

            a = n1 * rays.direction / n2
            a += self.diffraction_order * rays.wavelength * self.groove_normal(rays.position[x], rays.position[y])
            r = kgpy.vector.length(a)
            a = kgpy.vector.normalize(a)

            n = self.normal(rays.position[x], rays.position[y])
            c = -kgpy.vector.dot(a, n)

            p = -self.material.propagation_signum

            b = r * a + (r * c - p * np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * n

            rays.direction = kgpy.vector.normalize(b)
            rays.surface_normal = n
            if self.aperture is not None:
                if self.aperture.is_active:
                    rays.vignetted_mask &= self.aperture.is_unvignetted(rays.position)
            rays.index_of_refraction[...] = n2

        if not is_final_surface:
            rays = rays.copy()
            rays.position[z] -= self.thickness
            rays = rays.tilt_decenter(~self.transform_after)

        return rays

    def wavelength_from_angles(
            self,
            input_angle: u.Quantity,
            output_angle: u.Quantity,
    ) -> u.Quantity:
        a = np.sin(input_angle) + np.sin(output_angle)
        return a / (self.diffraction_order * self.groove_density)

import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import kgpy.vector
from .. import Rays
from . import coordinate, Surface, material as material_, aperture as aperture_

__all__ = ['Standard']

MaterialT = typ.TypeVar('MaterialT', bound=material_.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture_.Aperture)


@dataclasses.dataclass
class Standard(
    typ.Generic[MaterialT, ApertureT],
    Surface,
):

    radius: u.Quantity = np.inf * u.mm
    conic: u.Quantity = 0
    material: MaterialT = dataclasses.field(default_factory=lambda: material_.NoMaterial())
    aperture: ApertureT = dataclasses.field(default_factory=lambda: aperture_.NoAperture())
    transform_before: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter())
    transform_after: coordinate.TiltDecenter = dataclasses.field(default_factory=lambda: coordinate.TiltDecenter())

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'radius': self.radius,
            'conic': self.conic,
            'material': self.material.to_zemax(),
            'aperture': self.aperture.to_zemax(),
            'transform_before': self.transform_before,
            'transform_after': self.transform_after,
        })
        return args

    def to_zemax(self) -> 'Standard':
        from kgpy.optics import zemax
        return zemax.system.surface.Standard(**self.__init__args)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius,
            self.conic,
            self.material.config_broadcast,
            self.aperture.config_broadcast,
            self.transform_before.config_broadcast,
            self.transform_after.config_broadcast,
        )

    @property
    def curvature(self):
        return np.where(np.isinf(self.radius), 0, 1 / self.radius)

    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        r2 = np.square(x) + np.square(y)
        c = self.curvature
        return c * r2 / (1 + np.sqrt(1 - (1 + self.conic) * np.square(c) * r2))

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        x2 = np.square(x)
        y2 = np.square(y)
        c = self.curvature
        c2 = np.square(c)
        g = np.sqrt(1 - (1 + self.conic) * c2 * (x2 + y2))
        dzdx = c * x / g
        dzdy = c * y / g
        return kgpy.vector.normalize(kgpy.vector.from_components(dzdx, dzdy, -1 * u.dimensionless_unscaled))

    def propagate_rays(self, rays: Rays, is_first_surface: bool = False, is_final_surface: bool = False, ) -> Rays:

        if not is_first_surface:
            rays = rays.tilt_decenter(~self.transform_before)

            n1 = rays.index_of_refraction
            n2 = self.material.index_of_refraction(rays.wavelength, rays.polarization)

            a = rays.direction
            r = n1 / n2

            n = self.normal(rays.px, rays.py)
            c = -kgpy.vector.dot(a, n)

            p = self.material.propagation_signum

            b = r * a + (r * c - p * np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * n

            rays.position = self.calc_intercept(rays)
            rays.direction = b
            rays.surface_normal = n
            rays.vignetted_mask = self.aperture.check_points(rays.position)
            rays.index_of_refraction = n2

        if not is_final_surface:
            rays = rays.tilt_decenter(~self.transform_after)
            rays.pz -= self.thickness

        return rays

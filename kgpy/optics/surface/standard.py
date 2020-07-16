import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import OCC.Core.Geom
import OCC.Core.GeomAPI
import OCC.Core.GC
import OCC.Core.BRepPrimAPI
import OCC.Core.gp
import kgpy.optics.material.no_material
import kgpy.vector
from kgpy.vector import x, y, z
import kgpy.optics
from .. import Rays, coordinate, material as material_, aperture as aperture_
from . import Surface

__all__ = ['Standard']

MaterialT = typ.TypeVar('MaterialT', bound=typ.Optional[material_.Material])
ApertureT = typ.TypeVar('ApertureT', bound=typ.Optional[aperture_.Aperture])


@dataclasses.dataclass
class Standard(
    typ.Generic[MaterialT, ApertureT],
    Surface,
):

    radius: u.Quantity = np.inf * u.mm
    conic: float = 0
    material: MaterialT = None
    aperture: ApertureT = None
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
        out = np.broadcast(
            super().config_broadcast,
            self.radius,
            self.conic,
            self.transform_before.config_broadcast,
            self.transform_after.config_broadcast,
        )
        if self.material is not None:
            out = np.broadcast(out, self.material.config_broadcast)
        if self.aperture is not None:
            out = np.broadcast(out, self.aperture.config_broadcast)
        return out

    @property
    def is_plane(self):
        return self.radius == np.inf

    @property
    def is_hyperbola(self) -> np.ndarray:
        return self.conic < - 1

    @property
    def is_parabola(self) -> np.ndarray:
        return self.conic == 1

    @property
    def is_ellipse(self) -> np.ndarray:
        return -1 < self.conic < 0

    @property
    def is_sphere(self) -> np.ndarray:
        return self.conic == 0

    @property
    def is_oblate_ellisoid(self) -> np.ndarray:
        return self.conic > 0

    @property
    def curvature(self):
        return np.where(np.isinf(self.radius), 0, 1 / self.radius)

    def sag(self, ax: u.Quantity, ay: u.Quantity) -> u.Quantity:
        r2 = np.square(ax) + np.square(ay)
        c = self.curvature
        sz = c * r2 / (1 + np.sqrt(1 - (1 + self.conic) * np.square(c) * r2))
        mask = r2 >= np.square(self.radius)
        sz[mask] = 0
        return sz

    def normal(self, ax: u.Quantity, ay: u.Quantity) -> u.Quantity:
        x2 = np.square(ax)
        y2 = np.square(ay)
        c = self.curvature
        c2 = np.square(c)
        g = np.sqrt(1 - (1 + self.conic) * c2 * (x2 + y2))
        dzdx = c * ax / g
        dzdy = c * ay / g
        mask = (x2 + y2) >= np.square(self.radius)
        dzdx[mask] = 0
        dzdy[mask] = 0
        n = kgpy.vector.normalize(kgpy.vector.from_components(dzdx, dzdy, -1 * u.dimensionless_unscaled))
        return n

    def propagate_rays(self, rays: Rays, is_first_surface: bool = False, is_final_surface: bool = False, ) -> Rays:

        if not is_first_surface:
            rays = rays.tilt_decenter(~self.transform_before)
            rays.position = self.calc_intercept(rays)

            n1 = rays.index_of_refraction
            if self.material is not None:
                n2 = self.material.index_of_refraction(rays.wavelength, rays.polarization)
                p = rays.propagation_signum * self.material.propagation_signum
            else:
                n2 = 1 << u.dimensionless_unscaled
                p = rays.propagation_signum

            a = rays.direction
            r = n1 / n2

            n = self.normal(rays.position[x], rays.position[y])
            c = -kgpy.vector.dot(a, n)

            b = r * a + (r * c - p * np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * n

            rays.direction = kgpy.vector.normalize(b)
            rays.surface_normal = n
            if self.aperture is not None:
                if self.aperture.is_active:
                    rays.vignetted_mask = rays.vignetted_mask & self.aperture.is_unvignetted(rays.position)
            rays.index_of_refraction[...] = n2
            rays.propagation_signum = p

        if not is_final_surface:
            rays = rays.copy()
            rays.position[z] -= self.thickness
            rays = rays.tilt_decenter(~self.transform_after)

        return rays

    def apply_pre_transforms(self, value: u.Quantity, num_extra_dims: int = 0) -> u.Quantity:
        return self.transform_before(value, num_extra_dims=num_extra_dims)

    def apply_post_transforms(self, value: u.Quantity, num_extra_dims: int = 0) -> u.Quantity:
        value = self.transform_after(value, num_extra_dims=num_extra_dims)
        value[kgpy.vector.z] += self.thickness
        return value

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (0, 1),
            system: typ.Optional['kgpy.optics.System'] = None,
    ):
        if self.aperture is not None:
            self.aperture.plot_2d(ax, components, system, self)
        if self.material is not None:
            self.material.plot_2d(ax, components, system, self)

    @property
    def occ_surf(self) -> OCC.Core.Geom.Geom_Surface:

        if self.is_sphere:
            sphere_axis = OCC.Core.gp.gp_Ax3()
            occ_surf = OCC.Core.Geom.Geom_SphericalSurface(sphere_axis, self.radius.value)

        else:
            raise NotImplementedError

        return occ_surf









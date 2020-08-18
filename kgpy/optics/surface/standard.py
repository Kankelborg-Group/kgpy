import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
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
    """
    Most basic optical surface intended to be instantiated by the user.
    Allows for refractive or reflective surfaces with planar, spherical or conic figure.
    """
    radius: u.Quantity = np.inf * u.mm
    conic: u.Quantity = 0 * u.dimensionless_unscaled
    material: MaterialT = None
    aperture: ApertureT = None
    transform_before: typ.Optional[coordinate.TiltDecenter] = None
    transform_after: typ.Optional[coordinate.TiltDecenter] = None

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

    def to_occ(self):
        pass

    @property
    def config_broadcast(self):
        out = np.broadcast(
            super().config_broadcast,
            self.radius,
            self.conic,
        )
        if self.transform_before is not None:
            out = np.broadcast(out, self.transform_before.config_broadcast)
        if self.transform_after is not None:
            out = np.broadcast(out, self.transform_after.config_broadcast)
        if self.material is not None:
            out = np.broadcast(out, self.material.config_broadcast)
        if self.aperture is not None:
            out = np.broadcast(out, self.aperture.config_broadcast)
        return out

    @property
    def is_plane(self):
        return np.isinf(self.radius)

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
        return (self.conic == 0) & ~np.isinf(self.radius)

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

    def _index_of_refraction(self, rays: Rays) -> u.Quantity:
        """
        Index of refraction of this surface.
        Uses the index of refraction of the surface's material if available, otherwise returns 1.
        :param rays: Input rays to the surface. Needed if the index of refraction is wavelength / polarization dependent.
        :return: This surface's index of refraction.
        """
        if self.material is not None:
            return self.material.index_of_refraction(rays.wavelength, rays.polarization)
        else:
            return 1 << u.dimensionless_unscaled

    def _propagation_signum(self, rays: Rays) -> u.Quantity:
        p = np.sign(rays.direction[z])
        if self.material is not None:
            p = p * self.material.propagation_signum
        return p

    def _calc_input_direction(self, rays: Rays) -> u.Quantity:
        return rays.direction

    def _calc_index_ratio(self, rays: Rays) -> u.Quantity:
        n1 = rays.index_of_refraction
        n2 = self._index_of_refraction(rays)
        return n1 / n2

    def propagate_rays(self, rays: Rays, is_first_surface: bool = False, is_final_surface: bool = False, ) -> Rays:

        rays = rays.copy()

        if not is_first_surface:
            if self.transform_before is not None:
                rays = rays.tilt_decenter(~self.transform_before)
            rays.position = self.calc_intercept(rays)

            p = self._propagation_signum(rays)[..., None]
            a = self._calc_input_direction(rays)
            r = self._calc_index_ratio(rays)

            n = self.normal(rays.position[x], rays.position[y])
            c = -kgpy.vector.dot(a, n)

            b = r * a + (r * c - p * np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * n

            rays.direction = kgpy.vector.normalize(b)
            rays.surface_normal = n
            if self.aperture is not None:
                if self.aperture.is_active:
                    rays.vignetted_mask = rays.vignetted_mask & self.aperture.is_unvignetted(rays.position)
            rays.index_of_refraction[...] = self._index_of_refraction(rays)
            rays.propagation_signum = p

        if not is_final_surface:
            rays.position[z] -= self.thickness
            if self.transform_after is not None:
                rays = rays.tilt_decenter(~self.transform_after)

        return rays

    def apply_pre_transforms(self, value: u.Quantity, num_extra_dims: int = 0) -> u.Quantity:
        if self.transform_before is not None:
            value = self.transform_before(value, num_extra_dims=num_extra_dims)
        return value

    def apply_post_transforms(self, value: u.Quantity, num_extra_dims: int = 0) -> u.Quantity:
        if self.transform_after is not None:
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

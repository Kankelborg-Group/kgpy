import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
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
    transform_before: typ.Optional[coordinate.Transform] = None
    transform_after: typ.Optional[coordinate.Transform] = None
    intercept_error: u.Quantity = 0.1 * u.nm

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
        x2, y2 = np.square(ax), np.square(ay)
        c = self.curvature
        c2 = np.square(c)
        g = np.sqrt(1 - (1 + self.conic) * c2 * (x2 + y2))
        dzdx, dzdy = c * ax / g, c * ay / g
        mask = (x2 + y2) >= np.square(self.radius)
        dzdx[mask] = 0
        dzdy[mask] = 0
        n = kgpy.vector.normalize(kgpy.vector.from_components(dzdx, dzdy, -1 * u.dimensionless_unscaled))
        return n

    def ray_intercept(self, rays: Rays) -> u.Quantity:

        def line(t: u.Quantity) -> u.Quantity:
            return rays.position + rays.direction * t[..., None]

        def func(t: u.Quantity) -> u.Quantity:
            a = line(t)
            return a[z] - self.sag(a[x], a[y])

        bracket_max = 2 * np.nanmax(np.abs(self.rays_input.position[z])) + 1 * u.mm
        if np.isfinite(self.radius):
            bracket_max = np.sqrt(np.square(bracket_max) + 2 * np.square(self.radius))
        t_intercept = kgpy.optimization.root_finding.false_position(
            func=func,
            bracket_min=-bracket_max,
            bracket_max=bracket_max,
            max_abs_error=self.intercept_error,
        )
        return line(t_intercept)

    def _index_of_refraction(self, rays: Rays) -> u.Quantity:
        if self.material is not None:
            return self.material.index_of_refraction(rays)
        else:
            return np.sign(rays.index_of_refraction) << u.dimensionless_unscaled

    def _calc_input_direction(self, rays: Rays) -> u.Quantity:
        return rays.direction

    def _calc_index_ratio(self, rays: Rays) -> u.Quantity:
        n1 = rays.index_of_refraction
        n2 = self._index_of_refraction(rays)
        return n1 / n2

    @property
    def _rays_output(self) -> typ.Optional[Rays]:

        if self.rays_input is None:
            return None
        rays = self.rays_input.copy()

        rays.position = self.ray_intercept(rays)

        a = self._calc_input_direction(rays)
        r = self._calc_index_ratio(rays)

        n = self.normal(rays.position[x], rays.position[y])
        c = -kgpy.vector.dot(a, n)

        b = r * a + (r * c - np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * n

        rays.direction = kgpy.vector.normalize(b)
        rays.surface_normal = n
        if self.aperture is not None:
            if self.aperture.is_active:
                rays.vignetted_mask = rays.vignetted_mask & self.aperture.is_unvignetted(rays.position)
        rays.index_of_refraction[...] = self._index_of_refraction(rays)

        return rays

    @property
    def pre_transform(self) -> coordinate.TransformList:
        return coordinate.TransformList([self.transform_before])

    @property
    def post_transform(self) -> coordinate.TransformList:
        return coordinate.TransformList([self.transform_after, coordinate.Translate(z=self.thickness)])

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

    def plot_2d(
            self,
            ax: plt.Axes,
            transform: typ.Optional[coordinate.Transform] = None,
            components: typ.Tuple[int, int] = (0, 1),
    ):
        if self.aperture is not None:
            self.aperture.plot_2d(ax, self.sag, transform, components)
        if self.material is not None:
            self.material.plot_2d(ax, self.aperture, self.sag, transform, components)

    def copy(self) -> 'Standard':
        return Standard(
            name=self.name.copy(),
            thickness=self.thickness.copy(),
            is_active=self.is_active.copy(),
            is_visible=self.is_visible.copy(),
            radius=self.radius.copy(),
            conic=self.conic.copy(),
            material=self.material.copy(),
            aperture=self.aperture.copy(),
            transform_before=self.transform_before.copy(),
            transform_after=self.transform_after.copy(),
            intercept_error=self.intercept_error.copy(),
        )

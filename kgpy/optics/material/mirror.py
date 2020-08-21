import typing as typ
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.visualization
import kgpy.vector
import kgpy.optics
from .. import Rays
from . import Material


@dataclasses.dataclass
class Mirror(Material):
    thickness: u.Quantity = 0 * u.mm

    def to_zemax(self) -> 'Mirror':
        from kgpy.optics import zemax
        return zemax.system.surface.material.Mirror(thickness=self.thickness)

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return -np.sign(rays.index_of_refraction) * u.dimensionless_unscaled

    def copy(self) -> 'Mirror':
        return Mirror(
            thickness=self.thickness.copy(),
        )

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            surface: typ.Optional['kgpy.optics.surface.Standard'] = None,
    ):
        with astropy.visualization.quantity_support():

            c1, c2 = components
            wire = surface.aperture.wire.copy()
            wire[kgpy.vector.z] = self.thickness
            wire = surface.transform_to_global(wire, num_extra_dims=1)
            # wire = wire.reshape(wire.shape[:~2] + (wire.shape[~2] * wire.shape[~1], wire.shape[~0]))
            ax.fill(wire[..., c1].T, wire[..., c2].T, fill=False)

            if isinstance(surface.aperture, kgpy.optics.aperture.Polygon):

                front_vertices = surface.aperture.vertices.copy()
                back_vertices = surface.aperture.vertices.copy()
                front_vertices[kgpy.vector.z] = surface.sag(front_vertices[kgpy.vector.x], front_vertices[kgpy.vector.y])
                back_vertices[kgpy.vector.z] = self.thickness

                vertices = np.stack([front_vertices, back_vertices], axis=~1)
                vertices = surface.transform_to_global(vertices, num_extra_dims=2)
                vertices = vertices.reshape((-1, ) + vertices.shape[~1:])

                ax.plot(vertices[..., c1].T, vertices[..., c2].T, color='black')


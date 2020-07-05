import typing as typ
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.visualization
import kgpy.vector
import kgpy.optics
import kgpy.optics.surface
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

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            system: typ.Optional['kgpy.optics.System'] = None,
            surface: typ.Optional['kgpy.optics.surface.Standard'] = None,
    ):
        with astropy.visualization.quantity_support():

            c1, c2 = components
            edges = surface.aperture.edges.copy()
            edges[kgpy.vector.z] = self.thickness
            edges = surface.transform_to_global(edges, system, num_extra_dims=2)
            edges = edges.reshape(edges.shape[:~2] + (edges.shape[~2] * edges.shape[~1], edges.shape[~0]))
            ax.fill(edges[..., c1].T, edges[..., c2].T, fill=False)

            front_vertices = surface.aperture.vertices.copy()
            back_vertices = surface.aperture.vertices.copy()
            front_vertices[kgpy.vector.z] = surface.sag(front_vertices[kgpy.vector.x], front_vertices[kgpy.vector.y])
            back_vertices[kgpy.vector.z] = self.thickness

            vertices = np.stack([front_vertices, back_vertices], axis=~1)
            vertices = surface.transform_to_global(vertices, system, num_extra_dims=2)
            vertices = vertices.reshape((-1, ) + vertices.shape[~1:])

            ax.plot(vertices[..., c1].T, vertices[..., c2].T, color='black')


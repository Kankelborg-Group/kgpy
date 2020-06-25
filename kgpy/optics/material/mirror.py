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

            edges = surface.aperture.edges.copy()
            edges = edges.reshape(edges.shape[:~2] + (edges.shape[~2] * edges.shape[~1], edges.shape[~0]))
            # x_limits = edges[kgpy.vector.x].min(~0, keepdims=True), edges[kgpy.vector.x].max(~0, keepdims=True)
            # y_limits = edges[kgpy.vector.y].min(~0, keepdims=True), edges[kgpy.vector.y].max(~0, keepdims=True)
            front_vertices = surface.aperture.vertices.copy()
            back_vertices = surface.aperture.vertices.copy()
            front_vertices[kgpy.vector.z] = surface.sag(front_vertices[kgpy.vector.x], front_vertices[kgpy.vector.y])
            back_vertices[kgpy.vector.z] = self.thickness

            vertices = np.concatenate([front_vertices, back_vertices], axis=~1)

            vertices = surface.transform_to_global(vertices, system, num_extra_dims=1)

            ax.plot(vertices[..., c1].T, vertices[..., c2].T, color='black')

            # t = np.broadcast_to(self.thickness, x_limits[0].shape, subok=True)
            # for xlim in x_limits:
            #     for ylim in y_limits:
            #         z_edge = np.concatenate([
            #             np.stack([xlim, ylim, surface.sag(xlim, ylim)], axis=~0),
            #             np.stack([xlim, ylim, t], axis=~0),
            #         ], axis=~1)
            #         z_edge = surface.transform_to_global(z_edge, system, num_extra_dims=1)
            #         ax.plot(z_edge[..., c1].T, z_edge[..., c2].T, color='black')

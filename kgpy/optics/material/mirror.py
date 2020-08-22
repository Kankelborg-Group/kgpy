import typing as typ
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.visualization
from kgpy import vector, transform, optics
from kgpy.vector import x, y, z
from .. import Rays, Aperture
from . import Material


@dataclasses.dataclass
class Mirror(Material):
    thickness: u.Quantity = 0 * u.mm

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return -np.sign(rays.index_of_refraction) * u.dimensionless_unscaled

    def copy(self) -> 'Mirror':
        return Mirror(
            thickness=self.thickness.copy(),
        )

    def plot_2d(
            self,
            ax: plt.Axes,
            aperture: Aperture,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            rigid_transform: typ.Optional[transform.rigid.Transform] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
    ):
        with astropy.visualization.quantity_support():

            c1, c2 = components
            wire = aperture.wire.copy()
            wire[z] = self.thickness
            if transform is not None:
                wire = rigid_transform(wire, num_extra_dims=1)
            ax.fill(wire[..., c1].T, wire[..., c2].T, fill=False)

            # todo: generalize this for all aperture type
            if isinstance(aperture, optics.aperture.Polygon):

                front_vertices = aperture.vertices.copy()
                back_vertices = aperture.vertices.copy()
                front_vertices[z] = sag(front_vertices[x], front_vertices[y])
                back_vertices[z] = self.thickness

                vertices = np.stack([front_vertices, back_vertices], axis=~1)
                if transform is not None:
                    vertices = rigid_transform(vertices, num_extra_dims=2)
                vertices = vertices.reshape((-1, ) + vertices.shape[~1:])

                ax.plot(vertices[..., c1].T, vertices[..., c2].T, color='black')


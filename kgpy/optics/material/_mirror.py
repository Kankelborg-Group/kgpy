import typing as typ
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.visualization
from kgpy import vector, transform, optics
from kgpy.vector import x, y, z
from .. import Rays
from ..aperture import Aperture
from . import Material


@dataclasses.dataclass
class Mirror(Material):
    thickness: u.Quantity = 0 * u.mm

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return -np.sign(rays.index_of_refraction) * u.dimensionless_unscaled

    def copy(self) -> 'Mirror':
        other = super().copy()      # type: Mirror
        other.thickness = self.thickness

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            aperture: typ.Optional[Aperture] = None,
    ) -> plt.Axes:
        super().plot(ax=ax, components=components, rigid_transform=rigid_transform, sag=sag, aperture=aperture)

        if aperture is not None:
            with astropy.visualization.quantity_support():

                c1, c2 = components
                wire = aperture.wire.copy()
                wire[z] = self.thickness
                if rigid_transform is not None:
                    wire = rigid_transform(wire, num_extra_dims=1)
                ax.fill(wire[..., c1].T, wire[..., c2].T, fill=False)

                # todo: utilize polymorphsim here
                if isinstance(aperture, optics.aperture.Polygon):

                    front_vertices = aperture.vertices.copy()
                    back_vertices = aperture.vertices.copy()
                    front_vertices[z] = sag(front_vertices[x], front_vertices[y])
                    back_vertices[z] = self.thickness

                    vertices = np.stack([front_vertices, back_vertices], axis=~1)
                    if rigid_transform is not None:
                        vertices = rigid_transform(vertices, num_extra_dims=2)
                    vertices = vertices.reshape((-1, ) + vertices.shape[~1:])

                    ax.plot(vertices[..., c1].T, vertices[..., c2].T, color='black')

        return ax


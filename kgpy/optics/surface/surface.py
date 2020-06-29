import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.mixin
import kgpy.vector
import kgpy.optics
from .. import Rays, zemax_compatible

__all__ = ['Surface']


@dataclasses.dataclass
class Surface(
    zemax_compatible.ZemaxCompatible,
    zemax_compatible.InitArgs,
    kgpy.mixin.Broadcastable,
    kgpy.mixin.Named,
    abc.ABC
):
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    thickness: u.Quantity = 0 * u.mm
    is_active: 'np.ndarray[bool]' = np.array(True)
    is_visible: 'np.ndarray[bool]' = np.array(True)

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'thickness': self.thickness,
            'is_active': self.is_active,
            'is_visible': self.is_visible,
        })
        return args

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.thickness,
            self.is_active,
        )

    @property
    def thickness_vector(self):
        a = np.zeros(self.thickness.shape + (3,)) << self.thickness.unit
        a[..., ~0] = self.thickness
        return a

    def __iter__(self):
        yield self

    @abc.abstractmethod
    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        pass

    @abc.abstractmethod
    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        pass

    @abc.abstractmethod
    def propagate_rays(
            self,
            rays: Rays,
            is_first_surface: bool = False,
            is_final_surface: bool = False,
    ) -> Rays:
        pass

    def calc_intercept(
            self,
            rays: Rays,
            step_size: u.Quantity = 1 << u.m,
            max_error: u.Quantity = .1 << u.nm,
            max_iterations: int = 100,
    ) -> u.Quantity:

        intercept = rays.position.copy()

        t0 = -step_size
        t1 = step_size

        t0 = np.broadcast_to(t0, rays.wavelength.shape, subok=True)
        t1 = np.broadcast_to(t1, rays.wavelength.shape, subok=True)

        i = 0
        while True:

            if i > max_iterations:
                plt.show()
                raise ValueError('Number of iterations exceeded')

            a0 = intercept + rays.direction * t0
            a1 = intercept + rays.direction * t1
            plt.scatter(a1[kgpy.vector.x], a1[kgpy.vector.y])

            f0 = a0[kgpy.vector.z] - self.sag(a0[kgpy.vector.x], a0[kgpy.vector.y])
            f1 = a1[kgpy.vector.z] - self.sag(a1[kgpy.vector.x], a1[kgpy.vector.y])

            current_error = np.nanmax(np.abs(f1))
            print(current_error)
            if current_error < max_error:
                break

            f0 = np.expand_dims(f0, ~0)
            f1 = np.expand_dims(f1, ~0)

            m = (f1 - f0) == 0

            t2 = (t0 * f1 - t1 * f0) / (f1 - f0)
            t1 = np.broadcast_to(t1, t2.shape, subok=True)
            t2[m] = t1[m]

            t0 = t1
            t1 = t2

            i += 1

        t = t1
        intercept = intercept + rays.direction * t

        plt.show()

        return intercept

    @abc.abstractmethod
    def apply_pre_transforms(self, x: u.Quantity, num_extra_dims: int = 0) -> u.Quantity:
        return x

    @abc.abstractmethod
    def apply_post_transforms(self, x: u.Quantity, num_extra_dims: int = 0) -> u.Quantity:
        pass

    def transform_to_global(
            self, 
            x: u.Quantity, 
            system: typ.Optional['kgpy.optics.System'] = None, 
            num_extra_dims: int = 0
    ):

        if system is not None:
            surfaces = list(system)     # type: typ.List['Surface']
            index = None
            for s, surf in enumerate(surfaces):
                if surf is self:
                    index = s
                    break
            surfaces = surfaces[:index]
            surfaces.reverse()
    
            x = self.apply_pre_transforms(x, num_extra_dims)
    
            for surf in surfaces:
                x = surf.apply_pre_transforms(x, num_extra_dims)
                x = surf.apply_post_transforms(x, num_extra_dims)
    
        return x

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (0, 1),
            system: typ.Optional['kgpy.optics.System'] = None,
    ):
        pass






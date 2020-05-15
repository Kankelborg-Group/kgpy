import abc
import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import kgpy.mixin
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
            step_size: u.Quantity = 1 << u.mm,
            max_error: u.Quantity = 1 << u.nm,
            max_iterations: int = 100,
    ) -> u.Quantity:

        intercept = rays.position.copy()

        t0 = -step_size
        t1 = step_size

        i = 0
        while np.max(np.abs(t1 - t0)) < max_error:

            if i > max_iterations:
                raise ValueError('Number of iterations exceeded')

            a0 = intercept + rays.direction * t0
            a1 = intercept + rays.direction * t1

            f0 = a0[rays.z] - self.sag(a0[rays.x], a0[rays.y])
            f1 = a1[rays.z] - self.sag(a1[rays.x], a1[rays.y])

            t2 = (t0 * f1 - t1 * f0) / (f1 - f0)

            t0 = t1
            t1 = t2

            i += 1

        t = t1
        intercept += rays.direction * t

        return intercept

import abc
import dataclasses
import typing as typ
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.mixin
import kgpy.vector
from kgpy.vector import x, y, z
import kgpy.optimization.root_finding
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

    def index(self, surfaces: typ.Iterable['Surface']) -> int:
        for s, surf in enumerate(surfaces):
            if surf is self:
                return s

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
            bracket_min: u.Quantity,
            bracket_max: u.Quantity,
            max_error: u.Quantity = .1 << u.nm,
    ):
        def line(t: u.Quantity) -> u.Quantity:
            return rays.position + rays.direction * t[..., None]

        def func(t: u.Quantity) -> u.Quantity:
            a = line(t)
            return a[z] - self.sag(a[x], a[y])

        t_intercept = kgpy.optimization.root_finding.false_position(
            func=func,
            bracket_min=bracket_min,
            bracket_max=bracket_max,
            max_abs_error=max_error,
        )

        return line(t_intercept)

    @abc.abstractmethod
    def apply_pre_transforms(self, vector: u.Quantity, inverse: bool = False, num_extra_dims: int = 0) -> u.Quantity:
        pass

    @abc.abstractmethod
    def apply_post_transforms(self, vector: u.Quantity, inverse: bool = False, num_extra_dims: int = 0) -> u.Quantity:
        pass

    def transform_to_global(
            self, 
            vector: u.Quantity,
            system: typ.Optional['kgpy.optics.System'] = None, 
            num_extra_dims: int = 0
    ):

        if system is not None:
            surfaces = list(system)     # type: typ.List['Surface']
            index = surfaces.index(self)
            surfaces = surfaces[:index]
            surfaces.reverse()
    
            vector = self.apply_pre_transforms(vector, num_extra_dims=num_extra_dims)
    
            for surf in surfaces:
                vector = surf.apply_pre_transforms(vector, num_extra_dims=num_extra_dims)
                vector = surf.apply_post_transforms(vector, num_extra_dims=num_extra_dims)
    
        return vector

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (0, 1),
            system: typ.Optional['kgpy.optics.System'] = None,
    ):
        pass






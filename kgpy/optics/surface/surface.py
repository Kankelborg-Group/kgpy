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
from .. import coordinate, Rays, zemax_compatible

__all__ = ['Surface']


@dataclasses.dataclass
class Surface(
    zemax_compatible.ZemaxCompatible,
    zemax_compatible.InitArgs,
    kgpy.mixin.Copyable,
    kgpy.mixin.Broadcastable,
    kgpy.mixin.Named,
    abc.ABC
):
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    previous_surface: typ.Optional['Surface'] = dataclasses.field(default=None, compare=False, init=False)
    thickness: u.Quantity = 0 * u.mm
    is_active: 'np.ndarray[bool]' = np.array(True)
    is_visible: 'np.ndarray[bool]' = np.array(True)

    def __post_init__(self) -> typ.NoReturn:
        self.update()

    def update(self) -> typ.NoReturn:
        self._rays_output_cache = None

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
    def thickness_eff(self) -> u.Quantity:
        return self.thickness

    # def index(self, surfaces: typ.Iterable['Surface']) -> int:
    #     for s, surf in enumerate(surfaces):
    #         if surf is self:
    #             return s

    def __iter__(self):
        yield self

    @abc.abstractmethod
    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        pass

    @abc.abstractmethod
    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        pass

    @property
    def rays_input(self) -> typ.Optional[Rays]:
        rays = None
        if self.previous_surface is not None:
            rays = self.previous_surface.rays_output.apply_transform(self.previous_surface.post_transform)
        return rays

    @property
    @abc.abstractmethod
    def _rays_output(self) -> typ.Optional[Rays]:
        pass

    @property
    def rays_output(self) -> typ.Optional[Rays]:
        if self._rays_output_cache is None:
            self._rays_output_cache = self._rays_output
        return self._rays_output_cache

    @property
    @abc.abstractmethod
    def pre_transform(self) -> coordinate.Transform:
        pass

    @property
    @abc.abstractmethod
    def post_transform(self) -> coordinate.Transform:
        pass

    def transform_to_global(
            self, 
            vector: u.Quantity,
            num_extra_dims: int = 0
    ):
        if self.previous_surface is not None:
            vector = self.previous_surface.transform_to_global(vector, num_extra_dims=num_extra_dims)
            vector = self.previous_surface.post_transform(vector, num_extra_dims=num_extra_dims)
        return self.pre_transform(vector, num_extra_dims=num_extra_dims)

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (0, 1),
    ):
        pass

import typing as typ
import abc
import dataclasses
import numpy as np
import numpy.typing
import astropy.units as u
from kgpy import linspace, vector

__all__ = [
    'Grid1D',
    'Grid2D',
    'RegularGrid1D',
    'RegularGrid2D',
    'StratifiedRandomGrid1D',
    'StratifiedRandomGrid2D',
    'IrregularGrid1D',
]


class Grid1D(abc.ABC):
    pass

    @property
    @abc.abstractmethod
    def range(self) -> u.Quantity:
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> typ.Tuple[int, ...]:
        pass

    @property
    @abc.abstractmethod
    def points(self) -> u.Quantity:
        pass

    def mesh(self, shape: typ.Tuple[int, ...], axis: int) -> u.Quantity:
        sl = len(shape) * [np.newaxis]
        sl[axis] = slice(None)
        return np.broadcast_to(self.points[sl], shape)


class Grid2D(Grid1D):

    @property
    @abc.abstractmethod
    def range(self) -> vector.Vector2D:
        return super().range

    @property
    @abc.abstractmethod
    def points(self) -> vector.Vector2D:
        return super().points

    def mesh(self, shape: typ.Tuple[int, ...], axis: typ.Tuple[int, int]) -> vector.Vector2D:
        sl = len(shape) * [np.newaxis]
        sl[axis[0]] = slice(None)
        sl[axis[1]] = slice(None)
        return np.broadcast_to(self.points[sl], shape)


@dataclasses.dataclass
class RegularGrid1D(Grid1D):
    min: u.Quantity = 0 * u.dimensionless_unscaled
    max: u.Quantity = 0 * u.dimensionless_unscaled
    num_samples: int = 1

    @property
    def range(self) -> u.Quantity:
        return self.max - self.min

    @property
    def step_size(self) -> u.Quantity:
        return self.range / self.num_samples

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.range.shape + (self.num_samples, )

    @property
    def points(self) -> u.Quantity:
        return linspace(
            start=self.min,
            stop=self.max,
            num=self.num_samples,
            axis=~0
        )


@dataclasses.dataclass
class RegularGrid2D(RegularGrid1D, Grid2D):
    min: vector.Vector2D = dataclasses.field(default_factory=vector.Vector2D)
    max: vector.Vector2D = dataclasses.field(default_factory=vector.Vector2D)
    num_samples: vector.Vector2D = dataclasses.field(default_factory=lambda: vector.Vector2D(1, 1))

    @property
    def range(self) -> vector.Vector2D:
        return super().range

    @property
    def step_size(self) -> vector.Vector2D:
        return super().step_size

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.range.shape + self.num_samples.to_tuple()

    @property
    def points(self) -> vector.Vector2D:
        return vector.Vector2D(
            x=np.linspace(
                start=self.min.x,
                stop=self.max.x,
                num=self.num_samples.x,
                axis=~0
            ),
            y=np.linspace(
                start=self.min.y,
                stop=self.max.y,
                num=self.num_samples.y,
                axis=~0
            ),
        )


@dataclasses.dataclass
class StratifiedRandomGrid1D(RegularGrid1D):

    def perturbation(self, shape: typ.Tuple[int, ...]) -> u.Quantity:
        return self.step_size * (np.random.random_sample(shape) - 0.5)

    @property
    def points_base(self) -> u.Quantity:
        return super().points

    @property
    def points(self) -> u.Quantity:
        return self.points_base + self.perturbation(self.shape)

    def mesh(self, shape: typ.Tuple[int, ...], axis: int) -> u.Quantity:
        sl = len(shape) * [np.newaxis]
        sl[axis] = slice(None)
        return self.points_base[sl] + self.perturbation(shape=shape)


@dataclasses.dataclass
class StratifiedRandomGrid2D(RegularGrid2D, StratifiedRandomGrid1D):

    def perturbation(self, shape: typ.Tuple[int, ...]) -> vector.Vector2D:
        return vector.Vector2D(
            x=self.step_size.x * (np.random.random_sample(shape) - 0.5),
            y=self.step_size.y * (np.random.random_sample(shape) - 0.5),
        )

    def mesh(self, shape: typ.Tuple[int, ...], axis: typ.Tuple[int, int]) -> vector.Vector2D:
        sl = len(shape) * [np.newaxis]
        sl[axis[0]] = slice(None)
        sl[axis[1]] = slice(None)
        return self.points_base[sl] + self.perturbation(shape=shape)


@dataclasses.dataclass
class IrregularGrid1D(Grid1D):
    points: u.Quantity

    @property
    def range(self) -> u.Quantity:
        return self.points[..., ~0] - self.points[..., 0]

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.points.shape

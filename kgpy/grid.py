import typing as typ
import abc
import dataclasses
import numpy as np
import numpy.typing
import astropy.units as u
from kgpy import linspace, vector, mixin

__all__ = [
    'Grid1D',
    'Grid2D',
    'RegularGrid1D',
    'RegularGrid2D',
    'StratifiedRandomGrid1D',
    'StratifiedRandomGrid2D',
    'IrregularGrid1D',
]


@dataclasses.dataclass
class Grid1D(
    mixin.Copyable,
    abc.ABC,
):

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
        return np.broadcast_to(self.points[sl], shape, subok=True)


@dataclasses.dataclass
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
        sl_x = len(shape) * [np.newaxis]
        sl_y = len(shape) * [np.newaxis]
        sl_x[axis[0]] = slice(None)
        sl_y[axis[1]] = slice(None)
        points = self.points
        return vector.Vector2D(
            x=np.broadcast_to(points.x[sl_x], shape, subok=True),
            y=np.broadcast_to(points.y[sl_y], shape, subok=True),
        )


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

    def view(self) -> 'RegularGrid1D':
        other = super().view()  # type: RegularGrid1D
        other.min = self.min
        other.max = self.max
        other.num_samples = self.num_samples
        return other

    def copy(self) -> 'RegularGrid1D':
        other = super().copy()  # type: RegularGrid1D
        other.min = self.min.copy()
        other.max = self.max.copy()
        other.num_samples = self.num_samples
        return other


@dataclasses.dataclass
class RegularGrid2D(RegularGrid1D, Grid2D):
    min: vector.Vector2D = dataclasses.field(default_factory=vector.Vector2D)
    max: vector.Vector2D = dataclasses.field(default_factory=vector.Vector2D)
    num_samples: typ.Union[int, vector.Vector2D] = 1

    @property
    def num_samples_normalized(self) -> vector.Vector2D:
        num_samples = self.num_samples
        if not isinstance(num_samples, vector.Vector):
            num_samples = vector.Vector2D(x=num_samples, y=num_samples)
        return num_samples

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
                num=self.num_samples_normalized.x,
                axis=~0
            ),
            y=np.linspace(
                start=self.min.y,
                stop=self.max.y,
                num=self.num_samples_normalized.y,
                axis=~0
            ),
        )

    def view(self) -> 'RegularGrid2D':
        return super().view()

    def copy(self) -> 'RegularGrid2D':
        other = super().copy()  # type: RegularGrid2D
        if not isinstance(self.num_samples, int):
            other.num_samples = self.num_samples.copy()
        return other


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
        sl_x = len(shape) * [np.newaxis]
        sl_y = len(shape) * [np.newaxis]
        sl_x[axis[0]] = slice(None)
        sl_y[axis[1]] = slice(None)
        points = self.points_base
        perturbation = self.perturbation(shape=shape)
        return vector.Vector2D(
            x=points.x[sl_x] + perturbation.x,
            y=points.y[sl_y] + perturbation.y,
        )
        # return self.points_base[sl] + self.perturbation(shape=shape)


@dataclasses.dataclass
class IrregularGrid1D(Grid1D):
    points: u.Quantity = None

    @property
    def range(self) -> u.Quantity:
        return self.points[..., ~0] - self.points[..., 0]

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.points.shape

    def view(self) -> 'IrregularGrid1D':
        other = super().view()  # type: IrregularGrid1D
        other.points = self.points
        return other

    def copy(self) -> 'IrregularGrid1D':
        other = super().view()  # type: IrregularGrid1D
        other.points = self.points.copy()
        return other

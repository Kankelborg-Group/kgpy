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

    @property
    def name(self) -> typ.Optional[np.ndarray]:
        return None

    def mesh(self, shape: typ.Tuple[int, ...], new_axes: typ.Sequence[int]) -> u.Quantity:
        return np.expand_dims(self.points, axis=new_axes)


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

    def mesh(self, shape: typ.Tuple[int, ...], new_axes: typ.Sequence[int]) -> vector.Vector2D:
        points = self.points
        return vector.Vector2D(
            x=np.expand_dims(points.x[..., :, np.newaxis], axis=new_axes),
            y=np.expand_dims(points.y[..., np.newaxis, :], axis=new_axes),
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
        return self.range / (self.num_samples - np.array(1))

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
        return self.range.shape + self.num_samples_normalized.to_tuple()

    @property
    def points(self) -> vector.Vector2D:
        return vector.Vector2D(
            x=linspace(
                start=self.min.x,
                stop=self.max.x,
                num=self.num_samples_normalized.x,
                axis=~0
            ),
            y=linspace(
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
        rng = numpy.random.default_rng(42)
        return self.step_size * (rng.random(shape) - 0.5)

    @property
    def points_base(self) -> u.Quantity:
        return super().points

    @property
    def points(self) -> u.Quantity:
        return self.points_base + self.perturbation(self.shape)

    def mesh(self, shape: typ.Tuple[int, ...], new_axes: typ.Sequence[int]) -> u.Quantity:
        return super().mesh(shape=shape, new_axes=new_axes) + self.perturbation(shape=shape)


@dataclasses.dataclass
class StratifiedRandomGrid2D(
    RegularGrid2D,
    StratifiedRandomGrid1D
):

    @property
    def points_base(self) -> u.Quantity:
        return super().points

    def perturbation(self, shape: typ.Tuple[int, ...]) -> vector.Vector2D:
        rng = numpy.random.default_rng(42)
        return vector.Vector2D(
            x=self.step_size.x * (rng.random(shape) - 0.5),
            y=self.step_size.y * (rng.random(shape) - 0.5),
        )

    def mesh(self, shape: typ.Tuple[int, ...], new_axes: typ.Sequence[int]) -> vector.Vector2D:
        return super().mesh(shape=shape, new_axes=new_axes)


@dataclasses.dataclass
class IrregularGrid1D(Grid1D):
    points: u.Quantity = None
    name: typ.Optional[np.ndarray] = None

    @property
    def range(self) -> u.Quantity:
        return self.points[..., ~0] - self.points[..., 0]

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.points.shape

    def view(self) -> 'IrregularGrid1D':
        other = super().view()  # type: IrregularGrid1D
        other.points = self.points
        other.name = self.name
        return other

    def copy(self) -> 'IrregularGrid1D':
        other = super().view()  # type: IrregularGrid1D
        other.points = self.points.copy()
        if self.name is not None:
            other.name = self.name.copy()
        return other

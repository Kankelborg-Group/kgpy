import typing as typ
import abc
import dataclasses
import numpy as np
import numpy.typing
import astropy.units as u
from kgpy import linspace, vector, mixin
import kgpy.labeled

__all__ = [
    'Grid1D',
    'Grid2D',
    'RegularGrid1D',
    'RegularGrid2D',
    'StratifiedRandomGrid1D',
    'StratifiedRandomGrid2D',
    'IrregularGrid1D',
]

GridT = typ.TypeVar('GridT', bound='Grid')
OtherGridT = typ.TypeVar('OtherGridT', bound='Grid')
CoordinateT = typ.TypeVar('CoordinateT', bound=kgpy.labeled.AbstractArray)


@dataclasses.dataclass
class Grid(
    mixin.Copyable,
):

    @property
    def value(self: GridT) -> typ.Dict[str, kgpy.labeled.AbstractArray]:
        return vars(self)

    @value.setter
    def value(self: GridT, value: typ.Dict[str, kgpy.labeled.AbstractArray]):
        coordinates = self.value
        for axis in value:
            coordinates[axis] = value[axis]

    @property
    def components(self: GridT) -> typ.List[str]:
        return list(self.value.keys())

    @property
    def coordinates(self: GridT) -> typ.List[kgpy.labeled.AbstractArray]:
        return list(self.value.values())

    def __len__(self: GridT) -> int:
        return self.value.__len__()

    def __iter__(self: GridT) -> str:
        for axis in self.value:
            yield axis

    def __getitem__(self: GridT, axis: str) -> kgpy.labeled.AbstractArray:
        return self.value[axis]

    def __setitem__(self: GridT, axis: str, value: kgpy.labeled.AbstractArray):
        self.value[axis] = value

    @classmethod
    def from_dict(cls: typ.Type[GridT], value: typ.Dict[str, kgpy.labeled.AbstractArray]) -> GridT:
        self = cls()
        self.value = value
        return self

    def subspace(self: GridT, subgrid: OtherGridT) -> OtherGridT:
        other = type(subgrid)()
        for component in subgrid:
            other[component] = self[component]
        return other

    @property
    def shape(self: GridT) -> typ.Dict[str, int]:
        return kgpy.labeled.Array.broadcast_shapes(*self.coordinates)

    @property
    def ndim(self: GridT) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.array(self.shape.values()).prod()

    @property
    def broadcasted(self: GridT) -> GridT:
        shape = self.shape
        grid_broadcasted = type(self)()
        for component in self.value:
            setattr(grid_broadcasted, component, np.broadcast_to(getattr(self, component), shape=shape, subok=True))
        return grid_broadcasted

    def flatten(self: GridT, axis_new: str) -> GridT:
        other = self.broadcasted
        axes = other.shape.keys()
        for component in other:
            other[component] = other[component].combine_axes(axes=axes, axis_new=axis_new)
        return other

    def __eq__(self: GridT, other: OtherGridT) -> bool:
        for axis in self.value:
            if axis not in other.value:
                return False
            if not self.value[axis] == other.value[axis]:
                return False
        return True

    def view(self: GridT) -> 'GridT':
        other = super().view()
        for component in other:
            other[component] = self[component]
        return other

    def copy(self: GridT):
        other = super().copy()
        for component in other:
            other[component] = self[component].copy()
        return other

CoordinateX = typ.TypeVar('CoordinateX', bound=kgpy.labeled.AbstractArray)
CoordinateY = typ.TypeVar('CoordinateY', bound=kgpy.labeled.AbstractArray)
CoordinateZ = typ.TypeVar('CoordinateZ', bound=kgpy.labeled.AbstractArray)


@dataclasses.dataclass(eq=False)
class X(
    Grid,
    typ.Generic[CoordinateX],
):
    x: typ.Optional[CoordinateX] = None


@dataclasses.dataclass(eq=False)
class Y(
    Grid,
    typ.Generic[CoordinateY],
):
    y: typ.Optional[CoordinateY] = None


@dataclasses.dataclass(eq=False)
class Z(
    typ.Generic[CoordinateZ],
):
    z: typ.Optional[CoordinateZ] = None


@dataclasses.dataclass(eq=False)
class XY(Y, X):
    pass


@dataclasses.dataclass(eq=False)
class YZ(Z, Y):
    pass


@dataclasses.dataclass(eq=False)
class XZ(Z, X):
    pass


@dataclasses.dataclass(eq=False)
class XYZ(Z, Y, X):
    pass






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
    max: u.Quantity = 1 * u.dimensionless_unscaled
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
    max: vector.Vector2D = dataclasses.field(default_factory=lambda: 1 * u.dimensionless_unscaled + vector.Vector2D())
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

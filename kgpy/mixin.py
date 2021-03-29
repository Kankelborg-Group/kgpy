import abc
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.axes
import pandas
import pathlib
import pickle
import typing as typ

from kgpy import Name

__all__ = [
    'AutoAxis',
    'Broadcastable',
    'Named',
    'Dataframable',
    'Copyable',
    'Pickleable',
    'Plottable',
    'Toleranceable',
    'Colorable',
]


class AutoAxis:
    """
    Semi-automated axis numbering
    """

    @abc.abstractmethod
    def __init__(self):
        super().__init__()
        self.num_left_dim = 0       #: Number of dimensions on the left side of the array.
        self.num_right_dim = 0      #: Number of dimensions on the right side of the array
        self.all = []               #: List of indices for each axis.

    @property
    def ndim(self) -> int:
        return self.num_left_dim + self.num_right_dim

    def auto_axis_index(self, from_right: bool = True) -> int:
        if from_right:
            i = ~self.num_right_dim
            self.num_right_dim += 1
        else:
            i = self.num_left_dim
            self.num_left_dim += 1
        self.all.append(i)
        return i

    def perp_axes(self, axis: typ.Union[int, typ.Sequence[int]]) -> typ.Tuple[int, ...]:
        axes = self.all.copy()
        axes = [a % self.ndim for a in axes]
        if isinstance(axis, int):
            axis = [axis]
        for ax in axis:
            axes.remove(ax % self.ndim)
        return tuple([a - self.ndim for a in axes])


class Pickleable(abc.ABC):
    """
    Class for adding 'to_pickle' and 'from_pickle' methods for objects will long creation times.
    """

    @classmethod
    def from_pickle(cls, path: typ.Optional[pathlib.Path] = None):
        with open(path, 'rb') as file:
            self = pickle.load(file)
        return self

    def to_pickle(self, path: typ.Optional[pathlib.Path]):
        with open(path, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)


class Broadcastable:
    """
    Class to help with determining the shape of the optical configuration.
    In particular this class allows for cooperative subclassing by providing a default signature for the
    `config_broadcast` method.
    """

    @property
    def broadcasted(self):
        return np.broadcast()

    @property
    def shape(self):
        return self.broadcasted.shape


class Dataframable:
    """
    This mixin class naively converts a child class to a :class:`pandas.Dataframe`.
    """
    @property
    @abc.abstractmethod
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame()


class Copyable(abc.ABC):

    @abc.abstractmethod
    def view(self) -> 'Copyable':
        return type(self)()

    @abc.abstractmethod
    def copy(self) -> 'Copyable':
        return type(self)()


@dataclasses.dataclass
class Named(Copyable, Dataframable):
    name: Name = dataclasses.field(default_factory=lambda: Name())

    def view(self) -> 'Named':
        other = super().view()      # type: Named
        other.name = self.name
        return other

    def copy(self) -> 'Named':
        other = super().copy()     # type: Named
        other.name = self.name.copy()
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['name'] = [str(self.name)]
        return dataframe


class Toleranceable(abc.ABC):

    @property
    @abc.abstractmethod
    def tol_iter(self) -> typ.Iterator['Toleranceable']:
        yield self


@dataclasses.dataclass
class Colorable(Copyable):
    color: typ.Optional[str] = None

    def view(self) -> 'Colorable':
        other = super().view()  # type: Colorable
        other.color = self.color
        return other

    def copy(self) -> 'Colorable':
        other = super().copy()     # type: Colorable
        other.color = self.color
        return other


@dataclasses.dataclass
class Plottable(
    Colorable,
    abc.ABC
):
    linewidth: typ.Optional[float] = None
    linestyle: typ.Optional[str] = None

    @abc.abstractmethod
    def plot(
            self,
            ax: matplotlib.axes.Axes,
            components: typ.Tuple[str, ...],
            component_z: typ.Optional[str] = None,
            color: typ.Optional[str] = None,
            linewidth: typ.Optional[float] = None,
            linestyle: typ.Optional[str] = None,
            **kwargs,
    ) -> typ.List[matplotlib.lines.Line2D]:
        pass

    def view(self) -> 'Plottable':
        other = super().view()  # type: Plottable
        other.linewidth = self.linewidth
        other.linestyle = self.linestyle
        return other

    def copy(self) -> 'Plottable':
        other = super().copy()     # type: Plottable
        other.linewidth = self.linewidth
        other.linestyle = self.linestyle
        return other


ItemT = typ.TypeVar('ItemT')


@dataclasses.dataclass
class DataclassList(
    Toleranceable,
    Copyable,
    typ.Generic[ItemT],
):
    data: typ.List[ItemT] = dataclasses.field(default_factory=lambda: [])

    def __contains__(self, item: ItemT) -> bool:
        return self.data.__contains__(item)

    def __iter__(self) -> typ.Iterator[ItemT]:
        return self.data.__iter__()

    def __reversed__(self) -> typ.Iterator[ItemT]:
        return self.data.__reversed__()

    def __getitem__(self, item: typ.Union[int, slice]) -> ItemT:
        if isinstance(item, slice):
            other = self.view()
            other.data = self.data.__getitem__(item)
            return other
        else:
            return self.data.__getitem__(item)

    def __setitem__(self, key: typ.Union[int, slice], value: ItemT):
        self.data.__setitem__(key, value)

    def __delitem__(self, key: typ.Union[int, slice]):
        self.data.__delitem__(key)

    def __len__(self) -> int:
        return self.data.__len__()

    def __add__(self, other: 'DataclassList'):
        new_self = self.view()
        new_self.data = self.data.__add__(other.data)
        return new_self

    def index(self, value: ItemT) -> int:
        return self.data.index(value)

    def count(self, value: ItemT) -> int:
        return self.data.count(value)

    def append(self, item: ItemT) -> typ.NoReturn:
        self.data.append(item)

    def reverse(self):
        self.data.reverse()

    @classmethod
    def _tol_iter_data(cls, data: typ.List[ItemT]) -> typ.Iterator[typ.List[ItemT]]:
        if len(data) > 0:
            for d in data[0].tol_iter:
                for other_data in cls._tol_iter_data(data[1:]):
                    yield [d] + other_data
        else:
            yield data

    @property
    def tol_iter(self) -> typ.Iterator['DataclassList']:
        others = super().tol_iter   # type: typ.Iterator[DataclassList]
        for other in others:
            for data in self._tol_iter_data(self.data):
                new_other = other.view()
                new_other.data = data
                yield new_other

    def view(self) -> 'DataclassList':
        other = super().view()     # type: DataclassList
        other.data = self.data
        return other

    def copy(self) -> 'DataclassList':
        other = super().copy()      # type: DataclassList
        other.data = [d.copy() for d in self.data]
        return other


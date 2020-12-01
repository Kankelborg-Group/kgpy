import abc
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pathlib
import pickle
import typing as typ

from kgpy import Name

__all__ = ['AutoAxis', 'Broadcastable', 'Named', 'Dataframable', 'Copyable', 'Pickleable', 'Plottable']


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

    def auto_axis_index(self, from_right: bool = True):
        if from_right:
            i = ~self.num_right_dim
            self.num_right_dim += 1
        else:
            i = self.num_left_dim
            self.num_left_dim += 1
        self.all.append(i)
        return i

    def perp_axes(self, axis: int) -> typ.Tuple[int, ...]:
        axes = self.all.copy()
        axes = [a % self.ndim for a in axes]
        axes.remove(axis % self.ndim)
        return tuple([a - self.ndim for a in axes])


class Pickleable(abc.ABC):
    """
    Class for adding 'to_pickle' and 'from_pickle' methods for objects will long creation times.
    """

    @staticmethod
    @abc.abstractmethod
    def default_pickle_path() -> pathlib.Path:
        pass

    def to_pickle(self, path: typ.Optional[pathlib.Path] = None):
        if path is None:
            path = self.default_pickle_path()

        file = open(str(path), 'wb')
        pickle.dump(self, file)
        file.close()

        return

    @classmethod
    def from_pickle(cls, path: typ.Optional[pathlib.Path] = None):
        if path is None:
            path = cls.default_pickle_path()

        file = open(str(path), 'rb')
        self = pickle.load(file)
        file.close()

        return self


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
    def copy(self) -> 'Copyable':
        return type(self)()


@dataclasses.dataclass
class Named(Copyable, Dataframable):
    name: Name = dataclasses.field(default_factory=lambda: Name())

    def copy(self) -> 'Named':
        other = super().copy()     # type: Named
        other.name = self.name.copy()
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['name'] = [str(self.name)]
        return dataframe


class Plottable:

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
    ):
        if ax is None:
            fig, ax = plt.subplots()

        return ax

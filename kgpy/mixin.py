import abc
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pathlib
import pickle
import typing as typ

from kgpy import Name

__all__ = ['Broadcastable', 'Named', 'Dataframable', 'Copyable', 'Pickleable', 'Plottable']


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
    This mixin class naively converts a child class to a :py:class:`pandas.Dataframe`.
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

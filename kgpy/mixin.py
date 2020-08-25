import abc
import dataclasses
import numpy as np
import pandas

from .name import Name

__all__ = ['Broadcastable', 'Named', 'PandasDataframable', 'Copyable']


class Broadcastable:
    """
    Class to help with determining the shape of the optical configuration.
    In particular this class allows for cooperative subclassing by providing a default signature for the
    `config_broadcast` method.
    """

    @property
    def config_broadcast(self):
        return np.broadcast()

    @property
    def shape(self):
        return self.config_broadcast.shape


class PandasDataframable:
    """
    This mixin class naively converts a child class to a :py:class:`pandas.Dataframe`.
    """
    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(self.__dict__, orient='index')


class Copyable(abc.ABC):

    @abc.abstractmethod
    def copy(self) -> 'Copyable':
        return type(self)()


@dataclasses.dataclass
class Named(Copyable):
    name: Name = dataclasses.field(default_factory=lambda: Name())

    def copy(self) -> 'Named':
        other = super().copy()     # type: Named
        other.name = self.name.copy()
        return other

import typing as typ
import abc

__all__ = ['OCC_Compatible']


class OCC_Compatible(abc.ABC):

    @abc.abstractmethod
    def to_occ(self):
        pass
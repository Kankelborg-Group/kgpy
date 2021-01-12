import abc

__all__ = ['ZemaxCompatible']

import typing as typ


class ZemaxCompatible(abc.ABC):

    @abc.abstractmethod
    def to_zemax(self):
        pass


class InitArgs:

    @property
    def __init__args(self) -> typ.Dict['str', typ.Any]:
        return {}
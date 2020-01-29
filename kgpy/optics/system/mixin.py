import dataclasses
import typing as typ
import numpy as np
import kgpy.typing.numpy as npt 

__all__ = ['ConfigBroadcast', 'Named']


class ConfigBroadcast:
    """
    Class to help with determining the shape of the optical configuration.
    In particular this class allows for cooperative subclassing by providing a default signature for the
    `config_broadcast` method.
    """

    @property
    def config_broadcast(self):
        return np.broadcast()


@dataclasses.dataclass
class Named:
    """
    This class is useful if you want name to be the first argument in the constructor method signature.
    """
    name: str


ParentT = typ.TypeVar('ParentT')


@dataclasses.dataclass
class Adopted(typ.Generic[ParentT]):

    _parent: ParentT

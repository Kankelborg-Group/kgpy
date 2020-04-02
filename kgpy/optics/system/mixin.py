import dataclasses
import typing as typ
import numpy as np

from kgpy.optics.system.name import Name

__all__ = ['Broadcastable', 'Named']


class Broadcastable:
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
    name: Name = dataclasses.field(default_factory=lambda: Name())


ParentT = typ.TypeVar('ParentT')


@dataclasses.dataclass
class Adopted(typ.Generic[ParentT]):

    _parent: ParentT

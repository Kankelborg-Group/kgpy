import dataclasses
import numpy as np

from .name import Name

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

    @property
    def shape(self):
        return self.config_broadcast.shape


@dataclasses.dataclass
class Named:
    """ 
    This class is useful if you want name to be the first argument in the constructor method signature.
    """
    name: Name = dataclasses.field(default_factory=lambda: Name())

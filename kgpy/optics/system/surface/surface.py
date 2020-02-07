import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

from .. import mixin, coordinate

__all__ = ['Surface']


@dataclasses.dataclass
class Base(mixin.Named, mixin.ConfigBroadcast):

    _transform: coordinate.Transform = dataclasses.field(init=False, repr=False, default_factory=coordinate.Transform())

    thickness: u.Quantity = 0 * u.mm
    is_active: 'np.ndarray[bool]' = dataclasses.field(default_factory=lambda: np.array(True))

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.thickness,
            self.is_active,
        )

    @property
    def thickness_vector(self):
        a = np.zeros(self.thickness.shape + (3,)) << self.thickness.unit
        a[..., ~0] = self.thickness
        return a

    def __iter__(self):
        yield self


class Surface(Base):
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """
    
    @property
    def thickness(self) -> u.Quantity:
        return self._transform.translate.z

    @thickness.setter
    def thickness(self, value: u.Quantity):
        self._transform.translate.z = value

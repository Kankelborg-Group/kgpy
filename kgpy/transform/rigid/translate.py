import typing as typ
import dataclasses
import numpy as np
from astropy import units as u
import kgpy.vector
from . import Transform

__all__ = ['Translate']


@dataclasses.dataclass
class Translate(Transform):

    vector: u.Quantity = dataclasses.field(default_factory=kgpy.vector.from_components)

    @classmethod
    def from_components(cls, x: u.Quantity = 0 * u.mm, y: u.Quantity = 0 * u.mm, z: u.Quantity = 0 * u.mm):
        return cls(vector=kgpy.vector.from_components(x, y, z))

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.vector,
        )

    @property
    def x(self) -> u.Quantity:
        return self.vector[kgpy.vector.x]

    @x.setter
    def x(self, value: u.Quantity) -> typ.NoReturn:
        self.vector[kgpy.vector.x] = value

    @property
    def y(self) -> u.Quantity:
        return self.vector[kgpy.vector.y]

    @y.setter
    def y(self, value: u.Quantity) -> typ.NoReturn:
        self.vector[kgpy.vector.y] = value

    @property
    def z(self) -> u.Quantity:
        return self.vector[kgpy.vector.z]

    @z.setter
    def z(self, value: u.Quantity) -> typ.NoReturn:
        self.vector[kgpy.vector.z] = value

    def __invert__(self) -> 'Translate':
        return Translate(-self.vector)

    def __eq__(self, other: 'Translate') -> bool:
        return np.array(self.vector == other.vector).all()

    @property
    def rotation_eff(self) -> None:
        return None

    @property
    def translation_eff(self) -> u.Quantity:
        return self.vector

    def copy(self) -> 'Translate':
        return Translate(
            vector=self.vector.copy()
        )



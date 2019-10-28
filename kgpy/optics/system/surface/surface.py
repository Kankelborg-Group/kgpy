import dataclasses
import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u

__all__ = ['Surface']


@dataclasses.dataclass
class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    name: tp.Union[str, npt.Array[str]] = ''
    is_stop: tp.Union[bool, npt.Array[bool]] = False
    thickness: u.Quantity = 0 * u.mm

    @property
    def attributes(self):
        return [
            np.array(self.name),
            np.array(self.is_stop),
            self.thickness,
        ]

    @property
    def shape(self) -> tp.Tuple[int]:

        size_array = np.array([a.size for a in self.attributes])
        shape_array = [a.shape for a in self.attributes]

        return shape_array[np.argmax(size_array)]


import dataclasses
import numpy as np
import nptyping as npt
import astropy.units as u

__all__ = ['SurfaceArray']


@dataclasses.dataclass
class SurfaceArray:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    name: npt.Array[str] = np.array('')
    is_stop: npt.Array[bool] = np.array(False)
    thickness: u.Quantity = 0 * u.mm

import dataclasses
import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u

from . import Material, Aperture

__all__ = ['SurfaceArray']


@dataclasses.dataclass
class SurfaceArray:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    name: npt.Array[str] = np.array([['']])
    is_stop: npt.Array[bool] = np.array(False)
    thickness: u.Quantity = 0 * u.mm

    radius: u.Quantity = [[0]] * u.mm
    conic: u.Quantity = [[0]] * u.dimensionless_unscaled
    material: npt.Array[tp.Optional[Material]] = np.array([[None]])
    aperture: npt.Array[tp.Optional[Aperture]] = np.array([[None]])

    decenter_before: u.Quantity = [[[0, 0, 0]]] * u.m
    decenter_after: u.Quantity = [[[0, 0, 0]]] * u.m

    tilt_before: u.Quantity = [[[0, 0, 0]]] * u.deg
    tilt_after: u.Quantity = [[[0, 0, 0]]] * u.deg


import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

from kgpy import math

from . import Standard

__all__ = ['DiffractionGrating']

AperSurfT = typ.TypeVar('AperSurfT')
MainSurfT = typ.TypeVar('MainSurfT')


@dataclasses.dataclass
class DiffractionGrating(Standard[AperSurfT, MainSurfT]):

    diffraction_order: int = 1
    groove_frequency: u.Quantity = 0 * (1 / u.mm)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.diffraction_order,
            self.groove_frequency,
        )


def wavelength_from_vectors(input_vector: math.geometry.Vector,
                            output_vector: math.geometry.Vector,
                            grating_surface_normal: math.geometry.Vector,
                            grating_ruling_normal: math.geometry.Vector,
                            groove_frequency: u.Quantity,
                            diffraction_order: int
                            ) -> u.Quantity:

    r_hat = -grating_surface_normal
    z_hat = grating_ruling_normal

    phi_hat = z_hat.cross(r_hat)

    sin_alpha = input_vector.normalized.dot(phi_hat)
    sin_beta = output_vector.normalized.dot(phi_hat)

    wavelength = -(sin_alpha + sin_beta) / (diffraction_order * groove_frequency)

    return wavelength

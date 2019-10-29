import dataclasses
import typing as tp
import numpy as np
import astropy.units as u

from kgpy import math

from . import Standard

__all__ = ['DiffractionGrating']


@dataclasses.dataclass
class DiffractionGrating(Standard):

    diffraction_order: u.Quantity = 0 * u.dimensionless_unscaled
    groove_frequency: u.Quantity = 0 * (1 / u.mm)

    @property
    def broadcasted_attrs(self):
        return np.broadcast(
            super().broadcasted_attrs,
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

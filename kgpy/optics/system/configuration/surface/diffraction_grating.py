import dataclasses
import typing as tp
import astropy.units as u

from kgpy import math

from . import Standard

__all__ = ['DiffractionGrating']


class DiffractionGrating(Standard):

    def __init__(self, *args, diffraction_order: int = 1, groove_frequency: u.Quantity = 1000 / u.um, **kwargs):

        super().__init__(*args, **kwargs)

        self._diffraction_order = diffraction_order
        self._groove_frequency = groove_frequency

    @property
    def diffraction_order(self) -> int:
        return self._diffraction_order

    @property
    def groove_frequency(self) -> u.Quantity:
        return self._groove_frequency


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

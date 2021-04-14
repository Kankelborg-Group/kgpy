import typing as typ
import dataclasses
import scipy.optimize
import astropy.units as u

__all__ = ['Transmission']

import numpy as np


@dataclasses.dataclass
class Transmission:
    absorption_coefficient: u.Quantity
    scale_height: u.Quantity

    def optical_depth_vertical(self, observer_height: u.Quantity, ) -> u.Quantity:
        return self.absorption_coefficient * self.scale_height * np.exp(-observer_height / self.scale_height)

    def optical_depth(self, observer_height: u.Quantity, zenith_angle: u.Quantity, ) -> u.Quantity:
        return self.optical_depth_vertical(observer_height=observer_height) / np.cos(zenith_angle)

    def __call__(self, observer_height: u.Quantity, zenith_angle: u.Quantity, ) -> u.Quantity:
        return np.exp(-self.optical_depth(observer_height=observer_height, zenith_angle=zenith_angle))

    @classmethod
    def from_data_fit(
            cls,
            observer_height: u.Quantity,
            zenith_angle: u.Quantity,
            intensity_observed: u.Quantity,
            absorption_coefficient_guess: u.Quantity,
            scale_height_guess: u.Quantity,
            axis_fit: int = 0,
    ) -> 'Transmission':

        def factory(params: np.typing.ArrayLike):
            return cls(
                absorption_coefficient=params[0] * absorption_coefficient_guess.unit,
                scale_height=params[1] * scale_height_guess.unit,
            )

        def objective(params: np.typing.ArrayLike):
            self_test = factory(params=params)
            transmission = self_test(observer_height=observer_height, zenith_angle=zenith_angle)
            intensity_corrected = intensity_observed / transmission
            value = np.sqrt(np.mean(np.square(np.std(intensity_corrected, axis=axis_fit))))
            return value.value

        params_optimized = scipy.optimize.minimize(
            fun=objective,
            x0=np.array([
                absorption_coefficient_guess.value,
                scale_height_guess.value,
            ]),
            method='Nelder-Mead'
        ).x

        return factory(params_optimized)

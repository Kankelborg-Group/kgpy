import typing as typ
import dataclasses
import scipy.optimize
import scipy.signal
import scipy.special
import scipy.integrate
import astropy.units as u
import astropy.constants
from kgpy import vector, mixin

__all__ = ['Transmission']

import numpy as np


@dataclasses.dataclass
class Transmission:
    # location_base: vector.Vector2D
    # # pressure_gradient: vector.Vector2D
    # pressure_gradient: u.Quantity
    pressure_base: u.Quantity
    absorption_coefficient: u.Quantity
    scale_height: u.Quantity
    # scale_height_mesosphere: u.Quantity
    # scale_height_thermosphere: u.Quantity
    # height_mesopause: u.Quantity
    # scale_height_gradient: u.Quantity

    def optical_depth_vertical(
            self,
            # observer_latitude: u.Quantity,
            # observer_longitude: u.Quantity,
            observer_height: u.Quantity,
    ) -> u.Quantity:
        # dx = (observer_longitude - self.location_base.x)
        # dy = (observer_latitude - self.location_base.y)
        # px = self.pressure_gradient.x * dx
        # py = self.pressure_gradient.y * dy
        # px = 0 * u.cds.atm
        # py = self.pressure_gradient.y * dy + self.pressure_gradient.x * dy * dy
        pressure = self.pressure_base
        pressure = pressure.to(u.mbar)
        # pressure = pressure + self.pressure_gradient * np.sqrt(np.square(dx) + np.square(dy))
        scale_height = self.scale_height
        # scale_height = scale_height + self.scale_height_gradient * (observer_height - 250 * u.km)
        # scale_height = scale_height + px + py
        return pressure * self.absorption_coefficient * scale_height * np.exp(-observer_height / scale_height)

    def optical_depth(
            self,
            # observer_latitude: u.Quantity,
            # observer_longitude: u.Quantity,
            observer_height: u.Quantity,
            zenith_angle: u.Quantity,
    ) -> u.Quantity:
        optical_depth_vertical = self.optical_depth_vertical(
            # observer_latitude=observer_latitude,
            # observer_longitude=observer_longitude,
            observer_height=observer_height
        )
        return optical_depth_vertical / np.cos(zenith_angle)

    def __call__(
            self,
            # observer_latitude: u.Quantity,
            # observer_longitude: u.Quantity,
            observer_height: u.Quantity,
            zenith_angle: u.Quantity,
    ) -> u.Quantity:
        optical_depth = self.optical_depth(
            # observer_latitude=observer_latitude,
            # observer_longitude=observer_longitude,
            observer_height=observer_height,
            zenith_angle=zenith_angle,
        )
        return np.exp(-optical_depth)

    @classmethod
    def from_data_fit(
            cls,
            # location_base: vector.Vector2D,
            # observer_latitude: u.Quantity,
            # observer_longitude: u.Quantity,
            observer_height: u.Quantity,
            zenith_angle: u.Quantity,
            intensity_observed: u.Quantity,
            # pressure_gradient_guess: u.Quantity,
            # absorption_coefficient_guess: u.Quantity,
            # scale_height_guess: u.Quantity,
            # scale_height_gradient_guess: u.Quantity,
            pressure_base: u.Quantity,
            absorption_coefficient_bounds: u.Quantity,
            scale_height_bounds: u.Quantity,
            # scale_height_gradient_bounds: u.Quantity,
            axis_fit: int = 0,
    ) -> 'Transmission':
        def factory(params: np.typing.ArrayLike):
            (
                # pressure_gradient_x,
                # pressure_gradient_y,
                # pressure_gradient,
                absorption_coefficient,
                scale_height,
                # scale_height_gradient,
            ) = params
            return cls(
                # location_base=location_base,
                # pressure_gradient=vector.Vector2D(
                #     x=pressure_gradient_x * pressure_gradient_guess.x.unit,
                #     y=pressure_gradient_y * pressure_gradient_guess.y.unit,
                # ),
                # pressure_gradient=pressure_gradient << pressure_gradient_guess.unit,
                # pressure_gradient=pressure_gradient_guess,
                pressure_base=pressure_base,
                absorption_coefficient=absorption_coefficient * absorption_coefficient_bounds.unit,
                # scale_height=scale_height * scale_height_bounds.unit,
                scale_height=scale_height * scale_height_bounds.unit,
                # scale_height_gradient=scale_height_gradient * scale_height_gradient_bounds.unit,
                # scale_height_gradient=scale_height_gradient_guess,
            )

        def objective(params: np.typing.ArrayLike):
            self_test = factory(params=params)
            transmission = self_test(
                # observer_latitude=observer_latitude,
                # observer_longitude=observer_longitude,
                observer_height=observer_height,
                zenith_angle=zenith_angle,
            )
            intensity_corrected = intensity_observed / transmission
            intensity_corrected = scipy.signal.detrend(intensity_corrected, axis=axis_fit) << intensity_corrected.unit
            value = np.sqrt(np.mean(np.square(np.std(intensity_corrected, axis=axis_fit))))
            return value.value

        # params_optimized = scipy.optimize.minimize(
        #     fun=objective,
        #     x0=np.array([
        #         # pressure_gradient_guess.x.value,
        #         # pressure_gradient_guess.y.value,
        #         # pressure_gradient_guess.value,
        #         absorption_coefficient_guess.value,
        #         scale_height_guess.value,
        #         # scale_height_gradient_guess.value,
        #     ]),
        #     # method='Nelder-Mead'
        #     # method='Powell'
        # ).x

        params_optimized = scipy.optimize.brute(
            func=objective,
            ranges=[
                absorption_coefficient_bounds.value,
                # scale_height_bounds.value,
                scale_height_bounds.value,
                # scale_height_gradient_bounds.value,
            ],
            Ns=12,
        )

        return factory(params_optimized)


@dataclasses.dataclass
class TransmissionBates:
    # latitude_base: u.Quantity
    # longitude_base: u.Quantity
    density_base: u.Quantity
    # density_gradient_latitude: u.Quantity
    # density_gradient_longitude: u.Quantity
    absorption_coefficient: u.Quantity
    particle_mass: u.Quantity
    radius_base: u.Quantity
    scale_height: u.Quantity
    temperature_base: u.Quantity
    temperature_infinity: u.Quantity

    def temperature(self, radius: u.Quantity) -> u.Quantity:
        temp_change = self.temperature_infinity - self.temperature_base
        dr = radius - self.radius_base
        return self.temperature_infinity - temp_change * np.exp(-dr / self.scale_height)

    def temperature_gradient(self, radius: u.Quantity) -> u.Quantity:
        temp_change = self.temperature_infinity - self.temperature_base
        dr = radius - self.radius_base
        return temp_change * np.exp(-dr / self.scale_height) / self.scale_height

    @classmethod
    def gravitational_acceleration(cls, radius: u.Quantity):
        g = astropy.constants.G * astropy.constants.M_earth / np.square(radius + astropy.constants.R_earth)
        return g.to(u.m / u.s**2)

    def density(
            self,
            # latitude: u.Quantity,
            # longitude: u.Quantity,
            radius: u.Quantity,
            num_samples: int = 32,
    ) -> u.Quantity:

        m = self.particle_mass
        k = astropy.constants.k_B

        def integrand(r: u.Quantity) -> u.Quantity:
            g = self.gravitational_acceleration(r)
            T = self.temperature(r)
            dTdr = self.temperature_gradient(r)
            return (-g - (k / m) * dTdr) / T

        limit_lower = self.radius_base
        limit_upper = radius
        radius_steps = np.linspace(limit_lower, limit_upper, num_samples)
        integral = np.trapz(integrand(radius_steps), radius_steps, axis=0)

        # dx = longitude - self.longitude_base
        # dy = latitude - self.latitude_base
        # density_base = self.density_base + self.density_gradient_longitude * dx + self.density_gradient_latitude * dy
        density_base = self.density_base

        return density_base * np.exp((m / k) * integral)

    def optical_depth_vertical(
            self,
            # latitude: u.Quantity,
            # longitude: u.Quantity,
            radius: u.Quantity,
            num_samples: int = 32,
            num_samples_density: int = 32,
    ) -> u.Quantity:

        def integrand(
                # lat: u.Quantity,
                # lon: u.Quantity,
                r: u.Quantity
        ) -> u.Quantity:
            return self.density(
                # latitude=lat,
                # longitude=lon,
                radius=r,
                num_samples=num_samples_density,
            )

        limit_lower = radius
        limit_upper = radius + 500 * u.km
        radius_steps = np.linspace(limit_lower, limit_upper, num_samples)
        integral = np.trapz(integrand(
            # lat=latitude,
            # lon=longitude,
            r=radius_steps
        ), radius_steps, axis=0)

        return self.absorption_coefficient * integral

    def optical_depth(
            self,
            # latitude: u.Quantity,
            # longitude: u.Quantity,
            radius: u.Quantity,
            zenith_angle: u.Quantity,
            num_samples: int = 32,
            num_samples_density: int = 32,
    ) -> u.Quantity:
        optical_depth_vertical = self.optical_depth_vertical(
            # latitude=latitude,
            # longitude=longitude,
            radius=radius,
            num_samples=num_samples,
            num_samples_density=num_samples_density,
        )
        return optical_depth_vertical / np.cos(zenith_angle)

    def __call__(
            self,
            # latitude: u.Quantity,
            # longitude: u.Quantity,
            radius: u.Quantity,
            zenith_angle: u.Quantity,
            num_samples_depth: int = 32,
            num_samples_density: int = 32,
    ) -> u.Quantity:
        optical_depth = self.optical_depth(
            # latitude=latitude,
            # longitude=longitude,
            radius=radius,
            zenith_angle=zenith_angle,
            num_samples=num_samples_depth,
            num_samples_density=num_samples_density,
        )
        return np.exp(-optical_depth)

    @classmethod
    def from_data_fit(
            cls,
            # latitude: u.Quantity,
            # longitude: u.Quantity,
            observer_height: u.Quantity,
            zenith_angle: u.Quantity,
            intensity_observed: u.Quantity,
            # latitude_base: u.Quantity,
            # longitude_base: u.Quantity,
            density_base: u.Quantity,
            # density_gradient_latitude_bounds: u.Quantity,
            # density_gradient_latitude: u.Quantity,
            # density_gradient_longitude: u.Quantity,
            # density_gradient_longitude_bounds: u.Quantity,
            absorption_coefficient_bounds: u.Quantity,
            particle_mass: u.Quantity,
            radius_base: u.Quantity,
            scale_height_bounds: u.Quantity,
            temperature_base: u.Quantity,
            # temperature_base_bounds: u.Quantity,
            temperature_infinity_bounds: u.Quantity,
            axis_fit: int = 0,
            num_samples_depth: int = 32,
            num_samples_density: int = 32,
    ) -> 'TransmissionBates':
        def factory(params: np.typing.ArrayLike):
            (
                # density_gradient_latitude,
                # density_gradient_longitude,
                absorption_coefficient,
                scale_height,
                # temperature_base,
                temperature_infinity,
            ) = params
            return cls(
                # latitude_base=latitude_base,
                # longitude_base=longitude_base,
                density_base=density_base,
                # density_gradient_latitude=density_gradient_latitude,
                # density_gradient_latitude=density_gradient_latitude * density_gradient_latitude_bounds.unit,
                # density_gradient_longitude=density_gradient_longitude * density_gradient_longitude_bounds.unit,
                # density_gradient_longitude=density_gradient_longitude,
                absorption_coefficient=absorption_coefficient * absorption_coefficient_bounds.unit,
                particle_mass=particle_mass,
                radius_base=radius_base,
                scale_height=scale_height * scale_height_bounds.unit,
                temperature_base=temperature_base,
                # temperature_base=temperature_base * temperature_base_bounds.unit,
                temperature_infinity=temperature_infinity * temperature_infinity_bounds.unit,
            )

        def objective(params: np.typing.ArrayLike):
            self_test = factory(params=params)
            transmission = self_test(
                # latitude=latitude,
                # longitude=longitude,
                radius=observer_height,
                zenith_angle=zenith_angle,
                num_samples_depth=num_samples_depth,
                num_samples_density=num_samples_density,
            )
            intensity_corrected = intensity_observed / transmission
            # intensity_corrected = scipy.signal.detrend(intensity_corrected, axis=axis_fit) << intensity_corrected.unit
            value = np.sqrt(np.mean(np.square(np.std(intensity_corrected, axis=axis_fit))))
            # intensity_normalized = intensity_observed / intensity_observed.max(axis_fit)
            # intensity_normalized = intensity_normalized * transmission.max(axis_fit)
            # value = np.sqrt(np.mean(np.square(intensity_normalized - transmission)))
            return value.value

        params_optimized = scipy.optimize.brute(
            func=objective,
            ranges=[
                # density_gradient_latitude_bounds.value,
                # density_gradient_longitude_bounds.value,
                absorption_coefficient_bounds.value,
                scale_height_bounds.value,
                # temperature_base_bounds.value,
                temperature_infinity_bounds.value,
            ],
            Ns=11,
            # finish=None,
        )

        return factory(params_optimized)


@dataclasses.dataclass
class TransmissionList(mixin.DataclassList[Transmission]):

    def __call__(
            self,
            observer_height: u.Quantity,
            zenith_angle: u.Quantity,
    ) -> u.Quantity:
        transmission = 1 * u.dimensionless_unscaled
        for model in self:
            transmission = transmission * model(observer_height=observer_height, zenith_angle=zenith_angle)
        return transmission

    @classmethod
    def from_data_fit(
            cls,
            observer_height: u.Quantity,
            zenith_angle: u.Quantity,
            intensity_observed: u.Quantity,
            transmission_bounds: typ.Sequence[Transmission],
            axis_fit: int = 0,
    ) -> 'Transmission':
        def factory(params: np.typing.ArrayLike):
            # (
            #     absorption_coefficient_1,
            #     absorption_coefficient_2,
            #     scale_height_1,
            #     scale_height_2,
            # ) = params
            data = []
            params = params.reshape(-1, 2)
            for p, param in enumerate(params):
                (
                    absorption_coefficient,
                    scale_height,
                ) = param
                data.append(Transmission(
                    absorption_coefficient=absorption_coefficient * transmission_bounds[p].absorption_coefficient.unit,
                    scale_height=scale_height * transmission_bounds[p].scale_height.unit,
                ))
            return cls(data)

        def objective(params: np.typing.ArrayLike):
            self_test = factory(params=params)
            transmission = self_test(
                observer_height=observer_height,
                zenith_angle=zenith_angle,
            )
            intensity_corrected = intensity_observed / transmission
            intensity_corrected = scipy.signal.detrend(intensity_corrected, axis=axis_fit) << intensity_corrected.unit
            value = np.sqrt(np.mean(np.square(np.std(intensity_corrected, axis=axis_fit))))
            # print(self_test)
            # print(value)
            # print()
            return value.value

        # params_optimized = scipy.optimize.minimize(
        #     fun=objective,
        #     x0=np.array([
        #         # pressure_gradient_guess.x.value,
        #         # pressure_gradient_guess.y.value,
        #         # pressure_gradient_guess.value,
        #         absorption_coefficient_guess.value,
        #         scale_height_guess.value,
        #         # scale_height_gradient_guess.value,
        #     ]),
        #     # method='Nelder-Mead'
        #     # method='Powell'
        # ).x

        bounds = []
        for transmission_bound in transmission_bounds:
            bounds.append(transmission_bound.absorption_coefficient.value)
            bounds.append(transmission_bound.scale_height.value)

        params_optimized = scipy.optimize.brute(
            func=objective,
            ranges=bounds,
            Ns=7,
        )

        return factory(params_optimized)

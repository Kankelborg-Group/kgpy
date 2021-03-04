import typing as typ
import dataclasses
import numpy as np
import scipy.optimize
import astropy.units as u

__all__ = ['Logistic']


@dataclasses.dataclass
class Logistic:
    amplitude: u.Quantity = 1 * u.dimensionless_unscaled
    offset_x: u.Quantity = 0 * u.dimensionless_unscaled
    slope: u.Quantity = 1 * u.dimensionless_unscaled

    @classmethod
    def from_data_fit(
            cls,
            x: u.Quantity,
            y: u.Quantity,
            amplitude_guess: u.Quantity,
            offset_x_guess: u.Quantity,
            slope_guess: u.Quantity,
            axis_fit: int = 0,
    ):

        amplitude_unit = amplitude_guess.unit
        offset_x_unit = offset_x_guess.unit
        slope_unit = slope_guess.unit

        x, y = np.broadcast_arrays(x, y, subok=True)
        shape = x.shape

        x = np.moveaxis(x, axis_fit, 0)
        y = np.moveaxis(y, axis_fit, 0)

        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)

        amplitude = []
        offset_x = []
        slope = []

        for i in range(x.shape[~0]):
            # print(i)

            def objective(params: np.ndarray):
                self_test = cls(
                    amplitude=params[0] * amplitude_unit,
                    offset_x=params[1] * offset_x_unit,
                    slope=params[2] * slope_unit
                )
                residual = self_test(x[..., i]) - y[..., i]
                value = np.sqrt(np.mean(np.square(residual))).value
                # print(value)
                return value

            params_i = scipy.optimize.minimize(
                fun=objective,
                x0=np.array([amplitude_guess.value, offset_x_guess.value, slope_guess.value, ]),
                # method='Powell',
            ).x
            a_i, x0_i, k_i = params_i
            amplitude.append(a_i)
            offset_x.append(x0_i)
            slope.append(k_i)

        return cls(
            amplitude=(amplitude * amplitude_unit).reshape(shape[1:]),
            offset_x=(offset_x * offset_x_unit).reshape(shape[1:]),
            slope=(slope * slope_unit).reshape(shape[1:]),
        )

    def __call__(self, x: u.Quantity):
        return self.amplitude / (1 + np.exp(-self.slope * (x - self.offset_x)))

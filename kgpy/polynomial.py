import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import pandas
from kgpy import mixin, vector, format as fmt

__all__ = ['Polynomial3D']


@dataclasses.dataclass
class Polynomial3D(mixin.Dataframable):
    degree: int
    coefficients: typ.List[u.Quantity]
    component_names_input: typ.List[str] = dataclasses.field(default_factory=lambda: ['x', 'y', 'z'])
    component_names_output: typ.Optional[typ.List[str]] = None

    def __call__(self, x: u.Quantity, y: u.Quantity, z: u.Quantity):
        result = 0
        for c, v in zip(self.coefficients, self._vandermonde(x, y, z, degree=self.degree)):
            result += c * v[..., None]
        return result

    @classmethod
    def from_lstsq_fit(
            cls,
            x: u.Quantity,
            y: u.Quantity,
            z: u.Quantity,
            data: u.Quantity,
            mask: typ.Optional[np.ndarray] = None,
            degree: int = 1,
    ) -> 'Polynomial3D':

        if mask is None:
            mask = np.array([True])

        num_components_out = data.shape[~0:]
        grid_shape = np.broadcast(x, y, z, data[vector.x], mask).shape
        vgrid_shape = grid_shape + num_components_out
        shape = grid_shape[:~2]

        x = np.broadcast_to(x, grid_shape, subok=True)
        y = np.broadcast_to(y, grid_shape, subok=True)
        z = np.broadcast_to(z, grid_shape, subok=True)
        data = np.broadcast_to(data, vgrid_shape, subok=True)
        mask = np.broadcast_to(mask, grid_shape, subok=True)

        x = x.reshape(shape + (-1, ))
        y = y.reshape(shape + (-1, ))
        z = z.reshape(shape + (-1, ))
        data = data.reshape(shape + (-1, ) + num_components_out)
        mask = mask.reshape(shape + (-1, ))

        x = x.reshape((-1, ) + x.shape[~0:])
        y = y.reshape((-1, ) + y.shape[~0:])
        z = z.reshape((-1, ) + z.shape[~0:])
        data = data.reshape((-1,) + data.shape[~1:])
        mask = mask.reshape((-1, ) + mask.shape[~0:])

        coefficients = []
        for i in range(len(data)):
            m = mask[i]

            vander = cls._vandermonde(x=x[i][m], y=y[i][m], z=z[i][m], degree=degree)
            vander_value = [v.value for v in vander]
            vander_unit = [v.unit for v in vander]

            coeffs = np.linalg.lstsq(np.stack(vander_value, axis=~0), data[i][m, :])[0]

            coefficients.append([c / unit for c, unit in zip(coeffs, vander_unit)])

        coefficients = [u.Quantity(c) for c in zip(*coefficients)]
        coefficients = [c.reshape(shape + c.shape[~0:])[..., None, None, None, :] for c in coefficients]

        return Polynomial3D(
            degree=degree,
            coefficients=coefficients,
        )

    @staticmethod
    def _vandermonde(x: u.Quantity, y: u.Quantity, z: u.Quantity, degree: int = 1) -> typ.List[u.Quantity]:
        vander = []
        for d in range(degree + 1):
            for k in range(d + 1):
                for j in range(d + 1):
                    for i in range(d + 1):
                        if i + j + k == d:
                            vander.append((x ** i) * (y ** j) * (z ** k))
        # vander = []
        # if degree >= 0:
        #     vander += [np.ones(x.shape) * u.dimensionless_unscaled]
        # if degree >= 1:
        #     vander += [x, y, z]
        # if degree >= 2:
        #     vander += [x * x, x * y, y * y, x * z, y * z, z * z]
        # if degree >= 3:
        #     raise NotImplementedError
        return vander

    @property
    def coefficient_names(self):
        names = []
        for d in range(self.degree + 1):
            for k in range(d + 1):
                for j in range(d + 1):
                    for i in range(d + 1):
                        if i + j + k == d:
                            name_x = ''.join(self.component_names_input[vector.ix] * i) + ' '
                            name_y = ''.join(self.component_names_input[vector.iy] * j) + ' '
                            name_z = ''.join(self.component_names_input[vector.iz] * k) + ' '
                            name = '$C_{' + name_x + name_y + name_z + '}$'
                            names.append(name)
        return names

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        for coeff, name in zip(self.coefficients, self.coefficient_names):
            coeff = coeff[..., 0, 0, 0, :]
            if (np.abs(coeff.value) > 0.1).any():
                sci_notation = False
            else:
                sci_notation = True
            if np.isscalar(coeff):
                dataframe[name] = [fmt.quantity(coeff, scientific_notation=sci_notation)]
            else:
                dataframe[name] = [fmt.quantity(c, scientific_notation=sci_notation) for c in coeff.T.flatten()]
        if self.component_names_output is not None:
            dataframe.index = self.component_names_output

        return dataframe

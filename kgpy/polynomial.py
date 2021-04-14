import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import pandas
from kgpy import mixin, vector, format as fmt

__all__ = [
    'Polynomial3D',
    'Vector2DValuedPolynomial3D',
]


@dataclasses.dataclass
class Polynomial3D(mixin.Dataframable):
    degree: int
    coefficients: typ.List[u.Quantity]
    input_names: typ.Optional[typ.List[str]] = None
    output_name: typ.Optional[str] = None

    def __post_init__(self):
        if self.input_names is None:
            self.input_names = ['x', 'y', 'z']
        if self.output_name is None:
            self.output_name = 'f'

    def __call__(
            self,
            vector_input: vector.Vector3D,
            # x: u.Quantity, y: u.Quantity, z: u.Quantity
    ) -> u.Quantity:
        result = 0
        for c, v in zip(self.coefficients, self._vandermonde(vector_input, degree=self.degree)):
            result += c * v
        return result

    def dx(
            self,
            vector_input: vector.Vector3D,
    ) -> u.Quantity:
        result = 0
        for c, v in zip(self.coefficients, self._vandermonde_dx(vector_input, degree=self.degree)):
            result += c * v
        return result

    def dy(
            self,
            vector_input: vector.Vector3D,
    ) -> u.Quantity:
        result = 0
        for c, v in zip(self.coefficients, self._vandermonde_dy(vector_input, degree=self.degree)):
            result += c * v
        return result

    def dz(
            self,
            vector_input: vector.Vector3D,
    ) -> u.Quantity:
        result = 0
        for c, v in zip(self.coefficients, self._vandermonde_dz(vector_input, degree=self.degree)):
            result += c * v
        return result


    @classmethod
    def from_lstsq_fit(
            cls,
            # x: u.Quantity,
            # y: u.Quantity,
            # z: u.Quantity,
            # data: u.Quantity,
            data_input: vector.Vector3D,
            data_output: u.Quantity,
            mask: typ.Optional[np.ndarray] = None,
            degree: int = 1,
            input_names: typ.Optional[typ.List[str]] = None,
            output_name: typ.Optional[str] = None,
    ) -> 'Polynomial3D':

        if mask is None:
            mask = np.array([True])

        # num_components_out = data.shape[~0:]
        # grid_shape = np.broadcast(x, y, z, data[vector.x], mask).shape
        # vgrid_shape = grid_shape + num_components_out
        # grid_shape = np.broadcast(data_input, data_output, mask).shape
        # shape = grid_shape[:~2]

        # x = np.broadcast_to(x, grid_shape, subok=True)
        # y = np.broadcast_to(y, grid_shape, subok=True)
        # z = np.broadcast_to(z, grid_shape, subok=True)
        # data = np.broadcast_to(data, vgrid_shape, subok=True)
        # data_input = np.broadcast_to(data_input, grid_shape, subok=True)
        # data_output = np.broadcast_to(data_output, grid_shape, subok=True)
        # mask = np.broadcast_to(mask, grid_shape, subok=True)
        data_input, data_output, mask = np.broadcast_arrays(data_input, data_output, mask, subok=True)
        grid_shape = data_input.shape
        shape = grid_shape[:~2]

        # x = x.reshape(shape + (-1, ))
        # y = y.reshape(shape + (-1, ))
        # z = z.reshape(shape + (-1, ))
        # data = data.reshape(shape + (-1, ) + num_components_out)
        data_input = data_input.reshape(shape + (-1,))
        data_output = data_output.reshape(shape + (-1,))
        mask = mask.reshape(shape + (-1, ))

        # x = x.reshape((-1, ) + x.shape[~0:])
        # y = y.reshape((-1, ) + y.shape[~0:])
        # z = z.reshape((-1, ) + z.shape[~0:])
        # data = data.reshape((-1,) + data.shape[~1:])
        data_input = data_input.reshape((-1,) + data_input.shape[~0:])
        data_output = data_output.reshape((-1,) + data_output.shape[~0:])
        mask = mask.reshape((-1, ) + mask.shape[~0:])

        coefficients = []
        for i in range(len(data_output)):
            m = mask[i]

            vander = cls._vandermonde(vector_input=data_input[i][m], degree=degree)
            vander_value = [v.value for v in vander]
            vander_unit = [v.unit for v in vander]

            b = data_output[i][m]
            if isinstance(b, vector.Vector):
                b = b.quantity

            coeffs = np.linalg.lstsq(
                a=np.stack(vander_value, axis=~0),
                # b=data_output[i][m].quantity,
                b=b,
            )[0]

            coefficients.append([c / unit for c, unit in zip(coeffs, vander_unit)])

        if isinstance(data_output, vector.Vector):
            coefficients_factory = type(data_output).from_quantity
        else:
            coefficients_factory = lambda x: x

        coefficients = [coefficients_factory(u.Quantity(c)) for c in zip(*coefficients)]
        # coefficients = [c.reshape(shape + c.shape[~0:])[..., None, None, None, :] for c in coefficients]
        coefficients = [c.reshape(shape)[..., np.newaxis, np.newaxis, np.newaxis] for c in coefficients]

        return Polynomial3D(
            degree=degree,
            coefficients=coefficients,
            input_names=input_names,
            output_name=output_name,
        )

    @staticmethod
    def _vandermonde(vector_input: vector.Vector3D, degree: int = 1) -> typ.List[u.Quantity]:
        vander = []
        for d in range(degree + 1):
            for k in range(d + 1):
                for j in range(d + 1):
                    for i in range(d + 1):
                        if i + j + k == d:
                            vander.append((vector_input.x ** i) * (vector_input.y ** j) * (vector_input.z ** k))
        return vander

    @staticmethod
    def _vandermonde_dx(vector_input: vector.Vector3D, degree: int = 1) -> typ.List[u.Quantity]:
        vander = []
        for d in range(degree + 1):
            for k in range(d + 1):
                for j in range(d + 1):
                    for i in range(d + 1):
                        if i + j + k == d:
                            val_x = i * vector_input.x ** (i - 1)
                            val_y = vector_input.y ** j
                            val_z = vector_input.z ** k
                            val = val_x * val_y * val_z
                            val = np.nan_to_num(val)
                            vander.append(val)
        return vander

    @staticmethod
    def _vandermonde_dy(vector_input: vector.Vector3D, degree: int = 1) -> typ.List[u.Quantity]:
        vander = []
        for d in range(degree + 1):
            for k in range(d + 1):
                for j in range(d + 1):
                    for i in range(d + 1):
                        if i + j + k == d:
                            val_x = vector_input.x ** i
                            val_y = j * vector_input.y ** (j - 1)
                            val_z = vector_input.z ** k
                            val = val_x * val_y * val_z
                            val = np.nan_to_num(val)
                            vander.append(val)
        return vander

    @staticmethod
    def _vandermonde_dz(vector_input: vector.Vector3D, degree: int = 1) -> typ.List[u.Quantity]:
        vander = []
        for d in range(degree + 1):
            for k in range(d + 1):
                for j in range(d + 1):
                    for i in range(d + 1):
                        if i + j + k == d:
                            val_x = vector_input.x ** i
                            val_y = vector_input.y ** j
                            val_z = k * vector_input.z ** (k - 1)
                            val = val_x * val_y * val_z
                            val = np.nan_to_num(val)
                            vander.append(val)
        return vander

    @property
    def coefficient_names(self):
        names = []
        for d in range(self.degree + 1):
            for k in range(d + 1):
                for j in range(d + 1):
                    for i in range(d + 1):
                        if i + j + k == d:
                            name_x = ''.join(self.input_names[vector.ix] * i) + ' '
                            name_y = ''.join(self.input_names[vector.iy] * j) + ' '
                            name_z = ''.join(self.input_names[vector.iz] * k) + ' '
                            name = '$C_{' + name_x + name_y + name_z + '}$'
                            names.append(name)
        return names

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        for coeff, name in zip(self.coefficients, self.coefficient_names):
            coeff = coeff[..., 0, 0, 0]
            if (np.abs(coeff.value) > 0.1).any():
                sci_notation = False
            else:
                sci_notation = True
            if np.isscalar(coeff):
                dataframe[name] = [fmt.quantity(coeff, scientific_notation=sci_notation)]
            else:
                dataframe[name] = [fmt.quantity(c, scientific_notation=sci_notation) for c in coeff.T.flatten()]
        if self.output_name is not None:
            dataframe.index = [self.output_name]

        return dataframe

    def copy(self) -> 'Polynomial3D':
        input_names = self.input_names
        if input_names is not None:
            input_names = input_names.copy()
        return Polynomial3D(
            degree=self.degree,
            coefficients=[c.copy() for c in self.coefficients],
            input_names=input_names,
            output_name=self.output_name,
        )


@dataclasses.dataclass
class Vector2DValuedPolynomial3D(
    mixin.Dataframable,
):
    x: Polynomial3D
    y: Polynomial3D

    @classmethod
    def from_lstsq_fit(
            cls,
            data_input: vector.Vector3D,
            data_output: vector.Vector2D,
            mask: typ.Optional[np.ndarray] = None,
            degree: int = 1,
            input_names: typ.List[str] = None,
            output_names: typ.Optional[typ.List[str]] = None,
    ) -> 'Vector2DValuedPolynomial3D':
        if output_names is None:
            output_names = [None, None]
        return cls(
            x=Polynomial3D.from_lstsq_fit(
                data_input=data_input,
                data_output=data_output.x,
                mask=mask,
                degree=degree,
                input_names=input_names,
                output_name=output_names[vector.ix],
            ),
            y=Polynomial3D.from_lstsq_fit(
                data_input=data_input,
                data_output=data_output.y,
                mask=mask,
                degree=degree,
                input_names=input_names,
                output_name=output_names[vector.iy],
            ),
        )

    def __call__(
            self,
            vector_input: vector.Vector3D,
    ):
        return vector.Vector2D(
            x=self.x(vector_input=vector_input),
            y=self.y(vector_input=vector_input),
        )

    def dx(
            self,
            vector_input: vector.Vector3D,
    ):
        return vector.Vector2D(
            x=self.x.dx(vector_input=vector_input),
            y=self.y.dx(vector_input=vector_input),
        )

    def dy(
            self,
            vector_input: vector.Vector3D,
    ):
        return vector.Vector2D(
            x=self.x.dy(vector_input=vector_input),
            y=self.y.dy(vector_input=vector_input),
        )

    def dz(
            self,
            vector_input: vector.Vector3D,
    ):
        return vector.Vector2D(
            x=self.x.dz(vector_input=vector_input),
            y=self.y.dz(vector_input=vector_input),
        )

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.concat([self.x.dataframe, self.y.dataframe])




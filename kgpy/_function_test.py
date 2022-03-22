import typing as typ
import matplotlib.pyplot as plt
import time
import pytest
import cProfile
import numpy as np
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.function

__all__ = [
    'TestArray'
]


class TestArray:

    def test_shape(self):
        shape = dict(x=5, y=6)
        d = kgpy.function.Array(
            output=kgpy.labeled.Array(1, []),
            input=kgpy.vectors.Cartesian2D(
                x=kgpy.labeled.LinearSpace(start=0, stop=1, num=shape['x'], axis='x'),
                y=kgpy.labeled.LinearSpace(start=0, stop=2, num=shape['y'], axis='y'),
            ),
        )
        assert d.shape == shape

    def test_interp_nearest_1d(self):


        shape = dict(x=10, y=11)

        x = kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=shape['x'], axis='x')
        output = np.sin(x)
        # if width is not None:
        #     output = kgpy.uncertainty.Normal(output, width)
        a = kgpy.function.Array(
            input=x,
            output=output,
        )
        b = a.interp_nearest(x)

        assert np.all(a.output == b.output)

        x_fine = x.copy()
        x_fine.num = 100
        c = a.interp_nearest(x_fine)

        assert c.output.sum() != 0

        b.input = np.broadcast_to(b.input, b.input.shape)
        c.input = np.broadcast_to(c.input, c.input.shape)

        if isinstance(b.output, kgpy.uncertainty.AbstractArray):
            b.output = b.output.distribution.mean('_distribution')
        if isinstance(c.output, kgpy.uncertainty.AbstractArray):
            c.output = c.output.distribution.mean('_distribution')

        # plt.figure()
        # plt.scatter(a.input.array, a.output.array)
        # plt.scatter(b.input.array, b.output.array)
        # plt.scatter(c.input.array, c.output.array)
        # plt.show()

    @pytest.mark.parametrize(
        argnames='width',
        argvalues=[
            None,
            0.5,
        ]
    )
    def test_interp_nearest(self, width: typ.Optional[kgpy.labeled.ArrayLike]):

        shape = dict(x=10, y=11)

        x = kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=shape['x'], axis='x')
        y = kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=shape['y'], axis='y')
        position = kgpy.vectors.Cartesian2D(x, y)
        output = np.sin(x) * np.cos(y)
        if width is not None:
            output = kgpy.uncertainty.Normal(output, width)
        a = kgpy.function.Array(
            input=position,
            output=output,
        )
        b = a.interp_nearest(position)

        assert np.all(a.output == b.output)

        position_fine = kgpy.vectors.Cartesian2D(
            x=kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=100, axis='x'),
            y=kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=101, axis='y'),
        )
        c = a.interp_nearest(position_fine)

        assert c.output.sum() != 0

        b.input = np.broadcast_to(b.input, b.input.shape)
        c.input = np.broadcast_to(c.input, c.input.shape)

        if isinstance(b.output, kgpy.uncertainty.AbstractArray):
            b.output = b.output.distribution.mean('_distribution')
        if isinstance(c.output, kgpy.uncertainty.AbstractArray):
            c.output = c.output.distribution.mean('_distribution')

        # plt.figure()
        # plt.scatter(b.input.x.array, b.input.y.array, c=b.output.array)
        #
        # plt.figure()
        # plt.scatter(c.input.x.array, c.input.y.array, c=c.output.array)
        #
        # plt.show()

    def test_interp_linear_1d(self):

        shape = dict(x=10,)

        x = kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=shape['x'], axis='x')
        a = kgpy.function.Array(
            input=x,
            output=np.sin(x),
        )

        b = a.interp_linear(x)
        # assert np.isclose(a.data.data, b.data.data).all()

        shape_large = dict(x=100)
        x_large = kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=shape_large['x'], axis='x')
        c = a.interp_linear(
            x_large,
        )
        assert c.shape == shape_large

        # plt.figure()
        # plt.scatter(x=a.input.array, y=a.output.array)
        # plt.scatter(x=x.array,  y=b.array)
        # plt.scatter(x=x_large.array, y=c.array)
        # plt.show()

    def test_interp_linear_2d(self):

        shape = dict(
            x=10,
            y=11,
            z=12,
        )
        angle = 0.3
        x = kgpy.labeled.LinearSpace(start=-np.pi, stop=np.pi, num=shape['x'], axis='x')
        y = kgpy.labeled.LinearSpace(start=-np.pi, stop=np.pi, num=shape['y'], axis='y')
        z = kgpy.labeled.LinearSpace(start=0, stop=1, num=shape['z'], axis='z')
        x_rotated = x * np.cos(angle) - y * np.sin(angle)
        y_rotated = x * np.sin(angle) + y * np.cos(angle)
        position = kgpy.vectors.Cartesian2D(x, y)
        a = kgpy.function.Array(
            input=position,
            output=np.cos(x * x) * np.cos(y * y),
        )

        b = a.interp_linear(position)
        assert np.all(a.output - b < 1e-6)

        shape_large = dict(
            x=100,
            y=101,
            # z=shape['z'],
        )
        x_large = kgpy.labeled.LinearSpace(start=-np.pi, stop=np.pi, num=shape_large['x'], axis='x')
        y_large = kgpy.labeled.LinearSpace(start=-np.pi, stop=np.pi, num=shape_large['y'], axis='y')
        position_large = kgpy.vectors.Cartesian2D(x_large, y_large)
        c = a.interp_linear(position_large)

        assert c.shape == shape_large

        position = np.broadcast_to(position, position.shape)
        position_large = np.broadcast_to(position_large, position_large.shape)

        # plt.figure()
        # plt.scatter(
        #     x=position.x.array,
        #     y=position.y.array,
        #     c=a.output.array,
        #     # vmin=-1,
        #     # vmax=1,
        # )
        # plt.colorbar()
        #
        # plt.figure()
        # plt.scatter(
        #     x=position.x.array,
        #     y=position.y.array,
        #     c=b.array,
        # )
        # plt.colorbar()
        #
        # plt.figure()
        # plt.scatter(
        #     x=position_large.x.array,
        #     y=position_large.y.array,
        #     c=c.array,
        #     # vmin=-1,
        #     # vmax=1,
        # )
        # plt.colorbar()
        #
        # plt.show()

    def test_interp_barycentric_linear_1d(self):

        shape = dict(x=10,)

        x = kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=shape['x'], axis='x')
        a = kgpy.function.Array(
            input=kgpy.vectors.Cartesian1D(x),
            output=np.sin(x),
        )

        b = a.interp_barycentric_linear(input_new=kgpy.vectors.Cartesian1D(x))

        shape_large = dict(x=100)
        x_large = kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=shape_large['x'], axis='x')
        c = a.interp_barycentric_linear(input_new=kgpy.vectors.Cartesian1D(x_large))

        # fig, ax = plt.subplots()
        #
        # kgpy.vectors.Cartesian2D(a.input, a.output).scatter(ax, axis_plot='x', zorder=2)
        # kgpy.vectors.Cartesian2D(b.input, b.output).scatter(ax, axis_plot='x', zorder=1)
        # kgpy.vectors.Cartesian2D(c.input, c.output).scatter(ax, axis_plot='x', zorder=0)
        #
        # plt.show()

        assert np.isclose(a.output, b.output).all()

        assert c.shape == shape_large
        assert c.output.rms() > 0

    def test_interp_barycentric_linear_2d(self):

        shape = dict(
            x=14,
            y=14,
            # z=12,
        )
        angle = 0.3
        limit = np.pi
        x = kgpy.labeled.LinearSpace(start=-limit, stop=limit, num=shape['x'], axis='x')
        y = kgpy.labeled.LinearSpace(start=-limit, stop=limit, num=shape['y'], axis='y')
        # z = kgpy.labeled.LinearSpace(start=0, stop=1, num=shape['z'], axis='z')
        x_rotated = x * np.cos(angle) - y * np.sin(angle)
        y_rotated = x * np.sin(angle) + y * np.cos(angle)
        a = kgpy.function.Array(
            output=np.cos(x * x) * np.cos(y * y),
            input=kgpy.vectors.Cartesian2D(
                x=x_rotated,
                y=y_rotated,
                # z=z,
            )
        )

        b = a.interp_barycentric_linear(input_new=kgpy.vectors.Cartesian2D(
            x=x_rotated,
            y=y_rotated,
            # z=z,
        ))

        shape_large = dict(
            x=50,
            y=50,
            # z=shape['z'],
        )
        x_large = kgpy.labeled.LinearSpace(start=-limit, stop=limit, num=shape_large['x'], axis='x')
        y_large = kgpy.labeled.LinearSpace(start=-limit, stop=limit, num=shape_large['y'], axis='y')

        c = a.interp_barycentric_linear(kgpy.vectors.Cartesian2D(x_large, y_large))

        assert np.isclose(a.output, b.output).all()
        assert c.output.rms(where=np.isfinite(c.output)) > 0

    @pytest.mark.skip
    def test_interp_barycentric_linear_3d(self, capsys):

        capsys.disabled()

        samples = 30
        shape = dict(
            x=samples,
            y=samples,
            z=samples,
        )
        angle = 0.1
        limit = 10
        x = kgpy.labeled.LinearSpace(start=-limit, stop=limit, num=shape['x'], axis='x')
        y = LabeledArray.linspace(start=-limit, stop=limit, num=shape['y'], axis='y')
        z = LabeledArray.linspace(start=-limit, stop=limit, num=shape['z'], axis='z')
        x_rotated = x * np.cos(angle) * np.cos(angle) + y * np.cos(angle) * np.sin(angle) - z * np.sin(angle)
        y_rotated = -x * np.sin(angle) + y * np.cos(angle)
        z_rotated = x * np.cos(angle) * np.sin(angle) + y * np.sin(angle) * np.sin(angle) + z * np.cos(angle)
        a = DataArray(
            data=np.sin(x * y * z) / (x * y * z),
            grid=dict(
                x=x_rotated,
                # x=x,
                y=y_rotated,
                # y=y,
                z=z_rotated,
            )
        )

        b = a.interp_barycentric_linear(grid=dict(x=x_rotated, y=y_rotated, z=z_rotated))

        samples_large = 30
        shape_large = dict(
            x=samples_large,
            y=samples_large,
            z=samples_large,
        )
        x_large = LabeledArray.linspace(start=-2 * limit, stop=2 * limit, num=shape_large['x'], axis='x')
        y_large = LabeledArray.linspace(start=-2 * limit, stop=2 * limit, num=shape_large['y'], axis='y')
        z_large = LabeledArray.linspace(start=-2 * limit, stop=2 * limit, num=shape_large['z'], axis='z')

        profiler = cProfile.Profile()
        c = profiler.runcall(
            a.interp_barycentric_linear,
            # a.interp_barycentric_linear_scipy,
            grid=dict(
                x=x_large,
                y=y_large,
                z=z_large,
            ),
        )
        profiler.print_stats(sort='cumtime')

        profiler = cProfile.Profile()
        d = profiler.runcall(
            # a.interp_barycentric_linear,
            a.interp_barycentric_linear_scipy,
            grid=dict(
                x=x_large,
                y=y_large,
                z=z_large,
            ),
        )
        profiler.print_stats(sort='cumtime')

        mayavi.mlab.figure()
        mayavi.mlab.contour3d(
            a.grid_broadcasted['x'].data,
            a.grid_broadcasted['y'].data,
            a.grid_broadcasted['z'].data,
            a.data_broadcasted.data,
            opacity=0.5,
        )

        mayavi.mlab.figure()
        mayavi.mlab.contour3d(
            c.grid_broadcasted['x'].data,
            c.grid_broadcasted['y'].data,
            c.grid_broadcasted['z'].data,
            c.data_broadcasted.data,
            opacity=.5,
        )

        mayavi.mlab.figure()
        mayavi.mlab.contour3d(
            d.grid_broadcasted['x'].data,
            d.grid_broadcasted['y'].data,
            d.grid_broadcasted['z'].data,
            d.data_broadcasted.data,
            opacity=.5,
        )
        mayavi.mlab.figure()
        mayavi.mlab.contour3d(
            d.grid_broadcasted['x'].data,
            d.grid_broadcasted['y'].data,
            d.grid_broadcasted['z'].data,
            d.data_broadcasted.data - c.data_broadcasted.data,
            opacity=.5,
        )

        mayavi.mlab.show()

        assert np.isclose(a.data_broadcasted, b.data_broadcasted).data.all()


@pytest.mark.parametrize(
    argnames='coefficient_constant',
    argvalues=[
        1,
        kgpy.labeled.LinearSpace(0, 1, 7, axis='c'),
    ]
)
@pytest.mark.parametrize(
    argnames='coefficient_linear',
    argvalues=[
        2,
        kgpy.labeled.LinearSpace(0, 2, 7, axis='c'),
    ]
)
@pytest.mark.parametrize(
    argnames='coefficient_quadratic',
    argvalues=[
        3,
        kgpy.labeled.LinearSpace(0, 3, 7, axis='c'),
    ]
)
@pytest.mark.parametrize(
    argnames='distribution_width',
    argvalues=[
        None,
        1,
    ]
)
@pytest.mark.parametrize(
    argnames='unit_input_x',
    argvalues=[
        1,
        u.mm,
    ]
)
@pytest.mark.parametrize(
    argnames='unit_input_y',
    argvalues=[
        1,
        u.deg,
    ]
)
@pytest.mark.parametrize(
    argnames='unit_output',
    argvalues=[
        1,
        u.erg,
    ]
)
class TestPolynomial:

    def factory(
            self,
            coefficient_constant: kgpy.labeled.ArrayLike,
            coefficient_linear: kgpy.labeled.ArrayLike,
            coefficient_quadratic: kgpy.labeled.ArrayLike,
            distribution_width: kgpy.labeled.ArrayLike,
            unit_input_x: typ.Union[float, u.Unit],
            unit_input_y: typ.Union[float, u.Unit],
            unit_output: typ.Union[float, u.Unit],
    ) -> typ.Tuple[kgpy.function.PolynomialArray, kgpy.uncertainty.ArrayLike, kgpy.uncertainty.ArrayLike, kgpy.uncertainty.ArrayLike]:
        input = kgpy.vectors.Cartesian2D(
            x=kgpy.labeled.LinearSpace(0, 1, 11, axis='x') * unit_input_x,
            y=kgpy.labeled.LinearSpace(0, 1, 5, axis='y') * unit_input_y,
        )
        unit_input = kgpy.vectors.Cartesian2D(1 * unit_input_x, 1 * unit_input_y)
        coefficient_constant = coefficient_constant * unit_output
        coefficient_linear = coefficient_linear * unit_output / unit_input
        coefficient_quadratic = coefficient_quadratic * unit_output / unit_input ** 2

        if distribution_width is not None:
            coefficient_constant = kgpy.uncertainty.Uniform(coefficient_constant, width=distribution_width * unit_output)

        output = coefficient_quadratic * np.square(input) + coefficient_linear * input + coefficient_constant
        output = np.broadcast_to(output, output.shape).copy()
        input = np.broadcast_to(input, output.shape)
        mask = (input.x > 0.1 * unit_input_x) & (input.y > 0.1 * unit_input_y)
        output[~mask] = 100 * unit_output

        poly = kgpy.function.PolynomialArray(
            input=input,
            output=output,
            mask=mask,
            degree=2,
            axes_model=['x', 'y'],
        )

        return poly, coefficient_constant, coefficient_linear, coefficient_quadratic

    def test__call__(
            self,
            coefficient_constant: kgpy.labeled.ArrayLike,
            coefficient_linear: kgpy.labeled.ArrayLike,
            coefficient_quadratic: kgpy.labeled.ArrayLike,
            distribution_width: kgpy.labeled.ArrayLike,
            unit_input_x: typ.Union[float, u.Unit],
            unit_input_y: typ.Union[float, u.Unit],
            unit_output: typ.Union[float, u.Unit],
    ):
        poly, coefficient_constant, coefficient_linear, coefficient_quadratic = self.factory(
            coefficient_constant=coefficient_constant,
            coefficient_linear=coefficient_linear,
            coefficient_quadratic=coefficient_quadratic,
            distribution_width=distribution_width,
            unit_input_x=unit_input_x,
            unit_input_y=unit_input_y,
            unit_output=unit_output,
        )
        coefficients = poly.coefficients

        assert np.all(np.isclose(coefficients.coordinates[''].x, coefficient_constant))
        assert np.all(np.isclose(coefficients.coordinates['x'].x, coefficient_linear.x))
        assert np.all(np.isclose(coefficients.coordinates['x,x'].x, coefficient_quadratic.x))

        assert np.all(np.isclose(coefficients.coordinates[''].y, coefficient_constant))
        assert np.all(np.isclose(coefficients.coordinates['y'].y, coefficient_linear.y))
        assert np.all(np.isclose(coefficients.coordinates['y,y'].y, coefficient_quadratic.y))

        assert np.all(np.isclose(poly(poly.input), poly.output)[poly.mask])




import matplotlib.pyplot as plt
import time
import pytest
import cProfile
import numpy as np
import astropy.units as u
import mayavi.mlab
import kgpy.labeled
import kgpy.grid
import kgpy.data

__all__ = [
    'TestArray'
]


class TestArray:

    def test_shape(self):
        shape = dict(x=5, y=6)
        d = kgpy.data.Array(
            value=kgpy.labeled.Array(1, []),
            grid=kgpy.grid.XY(
                x=kgpy.labeled.LinearSpace(start=0, stop=1, num=shape['x'], axis='x'),
                y=kgpy.labeled.LinearSpace(start=0, stop=2, num=shape['y'], axis='y'),
            ),
        )
        assert d.shape == shape

    # def test_axes_separable(self):
    #     shape = dict(x=10, y=11, z=12)
    #     x = LabeledArray.linspace(0, 1, shape['x'], axis='x')
    #     y = LabeledArray.linspace(-1, 1, shape['y'], axis='y')
    #     a = DataArray(
    #         data=LabeledArray.empty(shape),
    #         grid=dict(
    #             x=x * y,
    #             y=y,
    #         ),
    #     )
    #     assert a.axes_separable == ['y', 'z']

    def test__eq__(self):
        shape = dict(x=10, y=11)
        start_x = 0
        stop_x = 1
        start_y = 0
        stop_y = 2
        a = kgpy.data.Array(
            value=kgpy.labeled.Array.ones(shape),
            grid=kgpy.grid.XY(
                x=kgpy.labeled.LinearSpace(start=start_x, stop=stop_x, num=shape['x'], axis='x'),
                y=kgpy.labeled.LinearSpace(start=start_y, stop=stop_y, num=shape['y'], axis='y'),
            ),
        )

        b = kgpy.data.Array(
            value=kgpy.labeled.Array.ones(shape),
            grid=kgpy.grid.XY(
                x=kgpy.labeled.LinearSpace(start=start_x, stop=stop_x, num=shape['x'], axis='x'),
                y=kgpy.labeled.LinearSpace(start=start_y, stop=stop_y, num=shape['y'], axis='y'),
            ),
        )

        assert a == b

    def test__eq__data(self):
        shape = dict(x=10, y=11)
        a = kgpy.data.Array(
            value=kgpy.labeled.Array.ones(shape),
            grid=kgpy.grid.XY(
                x=kgpy.labeled.LinearSpace(0, 1, num=shape['x'], axis='x'),
                y=kgpy.labeled.LinearSpace(0, 1, num=shape['y'], axis='y'),
            ),
        )

        b = kgpy.data.Array(
            value=kgpy.labeled.Array.zeros(shape),
            grid=kgpy.grid.XY(
                x=kgpy.labeled.LinearSpace(0, 1, num=shape['x'], axis='x'),
                y=kgpy.labeled.LinearSpace(0, 1, num=shape['y'], axis='y'),
            ),
        )

        assert a != b

    def test__eq__grid(self):
        shape = dict(x=10, y=11)
        a = kgpy.data.Array(
            value=kgpy.labeled.Array.ones(shape),
            grid=kgpy.grid.XY(
                x=kgpy.labeled.LinearSpace(0, 1, num=shape['x'], axis='x'),
                y=kgpy.labeled.LinearSpace(0, 1, num=shape['y'], axis='y'),
            ),
        )

        b = kgpy.data.Array(
            value=kgpy.labeled.Array.ones(shape),
            grid=kgpy.grid.XY(
                x=kgpy.labeled.LinearSpace(0, 2, num=shape['x'], axis='x'),
                y=kgpy.labeled.LinearSpace(0, 1, num=shape['y'], axis='y'),
            ),
        )

        assert a != b

    def test__getitem__int(self):

        shape = dict(x=5, y=6)

        x = kgpy.labeled.LinearSpace(start=0, stop=1, num=shape['x'], axis='x')
        y = kgpy.labeled.LinearSpace(start=0, stop=1, num=shape['y'], axis='y')

        a = kgpy.data.Array(
            value=x * y,
            grid=kgpy.grid.XY(x=x, y=y),
        )

        assert np.all(a[dict(y=~0)].value == x)
        assert np.all(a[dict(x=~0)].value == y)

        assert not np.all(a[dict(y=~1)].value == x)
        assert not np.all(a[dict(x=~1)].value == y)

    def test_interp_nearest(self):

        shape = dict(x=10, y=11)

        x = LabeledArray.linspace(start=0, stop=2 * np.pi, num=shape['x'], axis='x')
        y = LabeledArray.linspace(start=0, stop=2 * np.pi, num=shape['y'], axis='y')
        a = DataArray(
            data=np.sin(x) * np.cos(y),
            grid=dict(
                x=x,
                y=y,
            )
        )
        b = a.interp_nearest(x=x, y=y)

        assert a == b

        c = a.interp_nearest(
            x=LabeledArray.linspace(start=0, stop=2 * np.pi, num=100, axis='x'),
            y=LabeledArray.linspace(start=0, stop=2 * np.pi, num=101, axis='y'),
        )

        assert a != c

        plt.figure()
        plt.scatter(*np.meshgrid(a.grid['x'].data, a.grid['y'].data), c=a.data.data.T)

        plt.figure()
        plt.scatter(*np.meshgrid(c.grid['x'].data, c.grid['y'].data), c=c.data.data.T)

        # plt.show()

    def test_interp_linear_1d(self):

        shape = dict(x=10,)

        x = LabeledArray.linspace(start=0, stop=2 * np.pi, num=shape['x'], axis='x')
        a = DataArray(
            data=np.sin(x),
            grid=dict(x=x,)
        )

        b = a.interp_linear(x=x)
        assert np.isclose(a.data.data, b.data.data).all()

        shape_large = dict(x=100)
        c = a.interp_linear(
            x=LabeledArray.linspace(start=0, stop=2 * np.pi, num=shape_large['x'], axis='x'),
        )
        assert c.shape == shape_large

        # plt.figure()
        # plt.scatter(x=a.grid_broadcasted['x'].data, y=a.data_broadcasted.data)
        # plt.scatter(x=b.grid_broadcasted['x'].data,  y=b.data_broadcasted.data)
        # plt.scatter(x=c.grid_broadcasted['x'].data, y=c.data_broadcasted.data)
        # plt.show()

    def test_interp_linear_2d(self):

        shape = dict(
            x=10,
            y=11,
            z=12,
        )
        angle = 0.3
        x = LabeledArray.linspace(start=-np.pi, stop=np.pi, num=shape['x'], axis='x')
        y = LabeledArray.linspace(start=-np.pi, stop=np.pi, num=shape['y'], axis='y')
        z = LabeledArray.linspace(start=0, stop=1, num=shape['z'], axis='z')
        x_rotated = x * np.cos(angle) - y * np.sin(angle)
        y_rotated = x * np.sin(angle) + y * np.cos(angle)
        a = DataArray(
            data=np.cos(x * x) * np.cos(y * y),
            grid=dict(
                # x=x_rotated,
                x=x,
                # y=y_rotated,
                y=y,
                z=z,
            )
        )

        b = a.interp_linear(x=x, y=y)
        assert np.isclose(a.data, b.data).data.all()

        shape_large = dict(
            x=100,
            y=101,
            z=shape['z'],
        )
        x_large = LabeledArray.linspace(start=-np.pi, stop=np.pi, num=shape_large['x'], axis='x')
        y_large = LabeledArray.linspace(start=-np.pi, stop=np.pi, num=shape_large['y'], axis='y')
        c = a.interp_linear(
            x=x_large * np.cos(angle) - y_large * np.sin(angle),
            y=x_large * np.sin(angle) + y_large * np.cos(angle),
        )

        assert c.shape == shape_large

        # plt.figure()
        # plt.scatter(
        #     x=a.grid_broadcasted['x'].data,
        #     y=a.grid_broadcasted['y'].data,
        #     c=a.data_broadcasted.data,
        #     vmin=-1,
        #     vmax=1,
        # )
        # plt.colorbar()
        #
        # plt.figure()
        # plt.scatter(x=b.grid_broadcasted['x'].data, y=b.grid_broadcasted['y'].data,  c=b.data_broadcasted.data)
        # plt.colorbar()
        #
        # plt.figure()
        # plt.scatter(
        #     x=c.grid_broadcasted['x'].data,
        #     y=c.grid_broadcasted['y'].data,
        #     c=c.data_broadcasted.data,
        #     vmin=-1,
        #     vmax=1,
        # )
        # plt.colorbar()
        #
        # plt.show()

    def test_interp_barycentric_linear_1d(self):

        shape = dict(x=10,)

        x = kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=shape['x'], axis='x')
        a = kgpy.data.Array(
            value=np.sin(x),
            grid=kgpy.grid.X(x=x),
        )

        b = a.interp_barycentric_linear(grid=kgpy.grid.X(x))

        shape_large = dict(x=100)
        c = a.interp_barycentric_linear(
            grid=kgpy.grid.X(
                x=kgpy.labeled.LinearSpace(start=0, stop=2 * np.pi, num=shape_large['x'], axis='x'),
            ),
        )

        plt.figure()
        plt.scatter(x=c.grid.broadcasted.x.value, y=c.value_broadcasted.value)
        plt.scatter(x=b.grid.broadcasted.x.value,  y=b.value_broadcasted.value)
        plt.scatter(x=a.grid.broadcasted.x.value, y=a.value_broadcasted.value)
        plt.show()

        assert np.isclose(a.value.value, b.value.value).all()
        assert c.shape == shape_large

    def test_interp_barycentric_linear_2d(self):

        shape = dict(
            x=14,
            y=14,
            z=12,
        )
        angle = 0.3
        limit = np.pi
        x = kgpy.labeled.LinearSpace(start=-limit, stop=limit, num=shape['x'], axis='x')
        y = kgpy.labeled.LinearSpace(start=-limit, stop=limit, num=shape['y'], axis='y')
        z = kgpy.labeled.LinearSpace(start=0, stop=1, num=shape['z'], axis='z')
        x_rotated = x * np.cos(angle) - y * np.sin(angle)
        y_rotated = x * np.sin(angle) + y * np.cos(angle)
        a = kgpy.data.Array(
            value=np.cos(x * x) * np.cos(y * y),
            grid=kgpy.grid.XYZ(
                x=x_rotated,
                y=y_rotated,
                z=z,
            )
        )

        b = a.interp_barycentric_linear(grid=kgpy.grid.XY(
            x=x_rotated,
            y=y_rotated,
        ))

        shape_large = dict(
            x=50,
            y=50,
            z=shape['z'],
        )
        x_large = kgpy.labeled.LinearSpace(start=-limit, stop=limit, num=shape_large['x'], axis='x')
        y_large = kgpy.labeled.LinearSpace(start=-limit, stop=limit, num=shape_large['y'], axis='y')
        profiler = cProfile.Profile()
        c = profiler.runcall(
            a.interp_barycentric_linear,
            grid=kgpy.grid.XY(
                x=x_large,
                y=y_large,
            ),
        )
        profiler.print_stats(sort='cumtime')

        plt.figure()
        plt.scatter(
            x=a.grid.broadcasted.x.value,
            y=a.grid.broadcasted.y.value,
            c=a.value_broadcasted.value,
            vmin=-1,
            vmax=1,
        )
        plt.xlim([-limit, limit])
        plt.ylim([-limit, limit])
        plt.colorbar()

        plt.figure()
        plt.scatter(
            x=b.grid.broadcasted.x.value,
            y=b.grid.broadcasted.y.value,
            c=b.value_broadcasted.value,
            vmin=-1,
            vmax=1,
        )
        plt.xlim([-limit, limit])
        plt.ylim([-limit, limit])
        plt.colorbar()

        plt.figure()
        plt.scatter(
            x=b.grid.broadcasted.x.value,
            y=b.grid.broadcasted.y.value,
            c=(b.value - a.value).value,
            # vmin=-1,
            # vmax=1,
        )
        plt.xlim([-limit, limit])
        plt.ylim([-limit, limit])
        plt.colorbar()

        plt.figure()
        plt.scatter(
            x=c.grid.broadcasted.x.value,
            y=c.grid.broadcasted.y.value,
            c=c.value_broadcasted.value,
            vmin=-1,
            vmax=1,
        )
        plt.xlim([-limit, limit])
        plt.ylim([-limit, limit])
        plt.colorbar()

        plt.show()

        assert np.isclose(a.value_broadcasted, b.value_broadcasted).value.all()

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

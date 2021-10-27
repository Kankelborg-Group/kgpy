import matplotlib.pyplot as plt
import pytest
import numpy as np
from . import rebin, LabeledArray, DataArray


def test_rebin():
    x = np.random.rand(2, 5, 7, 3)
    scales = (2, 2, 2, 2)
    new_x = rebin(x, scales)
    assert new_x.shape == tuple(x.shape * np.array(scales))


class TestLabeledArray:

    def test__post_init__(self):
        with pytest.raises(ValueError):
            LabeledArray(data=np.empty((2, 3)), axis_names=['x'])

    def test_shape(self):
        shape = dict(x=2, y=3)
        a = LabeledArray(
            data=np.random.random(tuple(shape.values())),
            axis_names=list(shape.keys()),
        )
        assert a.shape == shape

    def test_shape_broadcasted(self):
        shape = dict(x=5, y=6)
        d1 = LabeledArray.empty(dict(x=shape['x'], y=1))
        d2 = LabeledArray.empty(dict(y=shape['y'], x=1))
        assert d1.shape_broadcasted(d2) == shape

    def test_data_aligned(self):
        shape = dict(x=5, y=6, z=7)
        d = LabeledArray.empty(dict(z=shape['z']))
        assert d._data_aligned(shape).shape == (1, 1, shape['z'])

    def test_linspace_scalar(self):
        shape = dict(x=100)
        d = LabeledArray.linspace(
            start=0,
            stop=1,
            num=shape['x'],
            axis='x',
        )
        assert d.shape == shape

    def test_combine_axes(self):
        shape = dict(x=5, y=6, z=7)
        a = LabeledArray.zeros(shape).combine_axes(['x', 'y'])
        assert a.shape == dict(z=shape['z'], xy=shape['x'] * shape['y'])

    def test__array_ufunc__(self):
        shape = dict(x=100, y=101)
        a = LabeledArray.linspace(
            start=0,
            stop=1,
            num=shape['x'],
            axis='x',
        )
        b = LabeledArray.linspace(
            start=0,
            stop=1,
            num=shape['y'],
            axis='y',
        )
        c = a + b
        assert c.shape == shape
        assert (c.data == a.data[..., np.newaxis] + b.data).all()

    def test__array_ufunc__incompatible_dims(self):
        a = LabeledArray.linspace(
            start=0,
            stop=1,
            num=10,
            axis='x',
        )
        b = LabeledArray.linspace(
            start=0,
            stop=1,
            num=11,
            axis='x',
        )
        with pytest.raises(ValueError):
            a + b

    def test__array_function__sum(self):
        shape = dict(x=4, y=7)
        a = np.sum(LabeledArray.ones(shape))
        assert a.data == shape['x'] * shape['y']
        assert a.shape == dict()

    def test__array_function__sum_axis(self):
        shape = dict(x=4, y=7)
        a = np.sum(LabeledArray.ones(shape), axis='x')
        assert (a.data == shape['x']).all()
        assert a.shape == dict(y=shape['y'])

    def test__array_function__sum_keepdims(self):
        shape = dict(x=4, y=7)
        a = np.sum(LabeledArray.ones(shape), keepdims=True)
        assert a.data[0, 0] == shape['x'] * shape['y']
        assert a.shape == dict(x=1, y=1)

    def test__getitem__int(self):
        a = LabeledArray.arange(stop=10, axis='x')
        b = LabeledArray.arange(stop=11, axis='y')
        c = LabeledArray.arange(stop=5, axis='z')
        d = a * b * c
        index = dict(x=1, y=1)
        assert (d[index].data == c.data).all()
        assert d[index].shape == c.shape

    def test__getitem__slice(self):
        a = LabeledArray.arange(stop=10, axis='x')
        b = LabeledArray.arange(stop=11, axis='y')
        c = LabeledArray.arange(stop=5, axis='z')
        d = a * b * c
        index = dict(x=slice(1, 2), y=slice(1, 2))
        assert (d[index].data == c.data).all()
        assert d[index].shape == dict(x=1, y=1, z=d.shape['z'])


class TestDataArray:

    def test_grid_normalized(self):
        shape = dict(x=5, y=6)
        d = DataArray(
            data=LabeledArray.ones(shape),
            grid=dict(
                y=LabeledArray.linspace(0, 1, shape['y'], axis='y'),
            ),
        )
        assert (d.grid_normalized['x'].data == np.arange(shape['x'])).all()

    def test_shape(self):
        shape = dict(x=5, y=6)
        d = DataArray(
            data=LabeledArray(1, []),
            grid=dict(
                x=LabeledArray.linspace(start=0, stop=1, num=shape['x'], axis='x'),
                y=LabeledArray.linspace(start=0, stop=2, num=shape['y'], axis='y'),
            )
        )
        assert d.shape == shape

    def test__eq__(self):
        shape = dict(x=10, y=11)
        a = DataArray(
            data=LabeledArray.ones(shape),
            grid=dict(
                x=LabeledArray.linspace(0, 1, shape['x'], axis='x'),
                y=LabeledArray.linspace(0, 1, shape['x'], axis='y'),
            ),
        )

        b = DataArray(
            data=LabeledArray.ones(shape),
            grid=dict(
                x=LabeledArray.linspace(0, 1, shape['x'], axis='x'),
                y=LabeledArray.linspace(0, 1, shape['x'], axis='y'),
            ),
        )

        assert a == b

    def test__eq__data(self):
        shape = dict(x=10, y=11)
        a = DataArray(
            data=LabeledArray.ones(shape),
            grid=dict(
                x=LabeledArray.linspace(0, 1, shape['x'], axis='x'),
                y=LabeledArray.linspace(0, 1, shape['y'], axis='y'),
            ),
        )

        b = DataArray(
            data=LabeledArray.zeros(shape),
            grid=dict(
                x=LabeledArray.linspace(0, 1, shape['x'], axis='x'),
                y=LabeledArray.linspace(0, 1, shape['y'], axis='y'),
            ),
        )

        assert a != b

    def test__eq__grid(self):
        shape = dict(x=10, y=11)
        a = DataArray(
            data=LabeledArray.ones(shape),
            grid=dict(
                x=LabeledArray.linspace(0, 1, shape['x'], axis='x'),
                y=LabeledArray.linspace(0, 1, shape['y'], axis='y'),
            ),
        )

        b = DataArray(
            data=LabeledArray.ones(shape),
            grid=dict(
                x=LabeledArray.linspace(0, 2, shape['x'], axis='x'),
                y=LabeledArray.linspace(0, 1, shape['y'], axis='y'),
            ),
        )

        assert a != b

    def test__getitem__int(self):

        shape = dict(x=5, y=6)

        x = LabeledArray.linspace(start=0, stop=1, num=shape['x'], axis='x')
        y = LabeledArray.linspace(start=0, stop=1, num=shape['y'], axis='y')

        a = DataArray(
            data=x * y,
            grid=dict(x=x, y=y),
        )

        assert np.all(a[dict(y=~0)].data == x)
        assert np.all(a[dict(x=~0)].data == y)

        assert not np.all(a[dict(y=~1)].data == x)
        assert not np.all(a[dict(x=~1)].data == y)


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



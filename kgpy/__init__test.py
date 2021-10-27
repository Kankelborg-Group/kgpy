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
            data=np.random.random(tuple(shape.values())),
            grid=dict(
                x=None,
                y=np.linspace(0, 1, shape['y']),
            ),
        )
        assert (d.grid_normalized['x'] == np.arange(shape['x'])[..., np.newaxis]).all()

    def test_shape_tuple(self):

        len_x = 6
        len_y = 7
        shape = dict(x=len_x, y=len_y)

        x, y = np.ix_(np.arange(len_x), np.linspace(0, 1, len_y))

        d = DataArray(data=1, grid=dict(x=x, y=y))

        assert d.shape == shape

    def test_get_item(self):

        shape = dict(x=5, y=6)
        shape_linear = list(shape.values())

        x, y = np.ix_(np.arange(shape['x']), np.linspace(0, 1, shape['y']))

        d = DataArray(
            np.random.random(shape_linear),
            grid=dict(
                x=x,
                y=y,
            )
        )

        assert d.get_item(x=0).shape == dict(y=shape['y'])
        assert d.get_item(y=~0).shape == dict(x=shape['x'])

        slice_test = slice(0, 3)
        assert d.get_item(x=slice_test).shape == dict(x=slice_test.stop, y=shape['y'],)
        assert d.get_item(y=slice_test).shape == dict(x=shape['x'], y=slice_test.stop,)

import pytest
import numpy as np
import astropy.units as u
import kgpy.labeled


class TestArray:

    def test__post_init__(self):
        with pytest.raises(ValueError):
            kgpy.labeled.Array(value=np.empty((2, 3)) * u.dimensionless_unscaled, axes=['x'])

    def test_shape(self):
        shape = dict(x=2, y=3)
        a = kgpy.labeled.Array(
            value=np.random.random(tuple(shape.values())) * u.dimensionless_unscaled,
            axes=['x', 'y'],
        )
        assert a.shape == shape

    def test_shape_broadcasted(self):
        shape = dict(x=5, y=6)
        d1 = kgpy.labeled.Array.empty(dict(x=shape['x'], y=1))
        d2 = kgpy.labeled.Array.empty(dict(y=shape['y'], x=1))
        assert d1.shape_broadcasted(d2) == shape

    def test_data_aligned(self):
        shape = dict(x=5, y=6, z=7)
        d = kgpy.labeled.Array.empty(dict(z=shape['z']))
        assert d._data_aligned(shape).shape == (1, 1, shape['z'])

    def test_combine_axes(self):
        shape = dict(x=5, y=6, z=7)
        a = kgpy.labeled.Array.zeros(shape).combine_axes(['x', 'y'])
        assert a.shape == dict(z=shape['z'], xy=shape['x'] * shape['y'])

    def test__array_ufunc__(self):
        shape = dict(x=100, y=101)
        a = kgpy.labeled.Array(
            value=np.random.random(shape['x']),
            axes=['x'],
        )
        b = kgpy.labeled.Array(
            value=np.random.random(shape['y']),
            axes=['y'],
        )
        c = a + b
        assert c.shape == shape
        assert (c.value == a.value[..., np.newaxis] + b.value).all()

    def test__array_ufunc__incompatible_dims(self):
        a = kgpy.labeled.Array(
            value=np.random.random(10),
            axes=['x'],
        )
        b = kgpy.labeled.Array(
            value=np.random.random(11),
            axes=['x'],
        )
        with pytest.raises(ValueError):
            a + b

    def test__array_function__sum(self):
        shape = dict(x=4, y=7)
        a = np.sum(kgpy.labeled.Array.ones(shape))
        assert a.value == shape['x'] * shape['y']
        assert a.shape == dict()

    def test__array_function__sum_axis(self):
        shape = dict(x=4, y=7)
        a = np.sum(kgpy.labeled.Array.ones(shape), axis='x')
        assert (a.value == shape['x']).all()
        assert a.shape == dict(y=shape['y'])

    def test__array_function__sum_keepdims(self):
        shape = dict(x=4, y=7)
        a = np.sum(kgpy.labeled.Array.ones(shape), keepdims=True)
        assert a.value[0, 0] == shape['x'] * shape['y']
        assert a.shape == dict(x=1, y=1)

    def test__getitem__int(self):
        a = kgpy.labeled.Range(stop=10, axis='x')
        b = kgpy.labeled.Range(stop=11, axis='y')
        c = kgpy.labeled.Range(stop=5, axis='z')
        d = a * b * c
        index = dict(x=1, y=1)
        assert (d[index].value == c.value).all()
        assert d[index].shape == c.shape

    def test__getitem__slice(self):
        a = kgpy.labeled.Range(stop=10, axis='x')
        b = kgpy.labeled.Range(stop=11, axis='y')
        c = kgpy.labeled.Range(stop=5, axis='z')
        d = a * b * c
        index = dict(x=slice(1, 2), y=slice(1, 2))
        assert (d[index].value == c.value).all()
        assert d[index].shape == dict(x=1, y=1, z=d.shape['z'])

    def test__getitem__advanced_bool(self):
        a = kgpy.labeled.Range(stop=10, axis='x')
        b = kgpy.labeled.Range(stop=11, axis='y')
        c = kgpy.labeled.Range(stop=5, axis='z')
        d = a * b * c
        assert d[a > 5].shape == {**b.shape, **c.shape, **d[a > 5].shape}

    def test_ndindex(self):
        shape = dict(x=2, y=2)
        result_expected = [{'x': 0, 'y': 0}, {'x': 0, 'y': 1}, {'x': 1, 'y': 0}, {'x': 1, 'y': 1}]
        assert list(kgpy.labeled.Array.ndindex(shape)) == result_expected
import pytest
import abc
import numpy as np
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty


class _TestAbstractArray(
    abc.ABC
):
    pass


class TestUniform(_TestAbstractArray):

    @pytest.mark.parametrize(
        argnames='nominal,width',
        argvalues=[
            (10, 1),
            (10 * u.mm, 1 * u.mm),
            (kgpy.labeled.Array(np.array([10, 11, 12]), axes=['x']), 1),
            (kgpy.labeled.Array(np.array([10, 11, 12]) * u.mm, axes=['x']), 1 * u.mm),
            (10, kgpy.labeled.Array(np.array([1, 2, 3]), axes=['x'])),
            (10 * u.mm, kgpy.labeled.Array(np.array([1, 2, 3]) * u.mm, axes=['x'])),
            (kgpy.labeled.Array(np.array([10, 11, 12]) * u.mm, axes=['x']), kgpy.labeled.Array(np.array([1, 2, 3]) * u.mm, axes=['y'])),
        ],
    )
    def test_distribution(self, nominal: kgpy.labeled.ArrayLike, width: kgpy.labeled.ArrayLike):
        a = kgpy.uncertainty.Uniform(nominal=nominal, width=width, num_samples=11)
        assert isinstance(a, kgpy.uncertainty.Uniform)
        assert isinstance(a.distribution, kgpy.labeled.UniformRandomSpace)
        if not isinstance(nominal, kgpy.labeled.AbstractArray):
            nominal = kgpy.labeled.Array(nominal)
        assert np.all(a.distribution.min(axis='_distribution') >= nominal - width)


class TestNormal(_TestAbstractArray):

    pass


class TestArray:

    @pytest.mark.parametrize(
        argnames='a,b',
        argvalues=[
            (kgpy.uncertainty.Uniform(10, width=1), 5),
            (kgpy.uncertainty.Uniform(10, width=1), kgpy.labeled.LinearSpace(5, 6, 9, axis='b')),
            (kgpy.uncertainty.Uniform(kgpy.labeled.LinearSpace(10, 11, 7, axis='a'), width=1), 5),
            (kgpy.uncertainty.Uniform(kgpy.labeled.LinearSpace(10, 11, 7, axis='a'), width=1), kgpy.labeled.LinearSpace(5, 6, 9, axis='b')),
            (kgpy.uncertainty.Uniform(10 * u.mm, width=1*u.mm), 5 * u.mm),
            (kgpy.uncertainty.Uniform(10 * u.mm, width=1 * u.mm), kgpy.labeled.LinearSpace(5, 6, 9, axis='b') * u.mm),
            (kgpy.uncertainty.Uniform(kgpy.labeled.LinearSpace(10, 11, 7, axis='a') * u.mm, width=1 * u.mm), 5 * u.mm),
            (kgpy.uncertainty.Uniform(kgpy.labeled.LinearSpace(10, 11, 7, axis='a') * u.mm, width=1 * u.mm), kgpy.labeled.LinearSpace(5, 6, 9, axis='b') * u.mm)
        ],
    )
    def test__add__(self, a, b):
        c = a + b
        d = b + a
        b_normalized = b
        if not isinstance(b, kgpy.uncertainty.AbstractArray):
            b_normalized = kgpy.uncertainty.Array(b)
        assert isinstance(c, kgpy.uncertainty.AbstractArray)
        assert isinstance(d, kgpy.uncertainty.AbstractArray)
        assert np.all(c.nominal == a.nominal + b_normalized.nominal)
        assert np.all(d.nominal == b_normalized.nominal + a.nominal)
        assert np.all(c == d)

    @pytest.mark.parametrize(
        argnames='a,b',
        argvalues=[
            (kgpy.uncertainty.Uniform(10, width=1), u.mm),
            (kgpy.uncertainty.Uniform(10, width=1), 5),
            (kgpy.uncertainty.Uniform(10, width=1), kgpy.labeled.LinearSpace(5, 6, 9, axis='b')),
            (kgpy.uncertainty.Uniform(10, width=1), kgpy.uncertainty.Uniform(10, width=2)),
        ]
    )
    def test__mul__(self, a, b):
        c = a * b
        d = b * a
        assert isinstance(c, kgpy.uncertainty.AbstractArray)
        assert isinstance(d, kgpy.uncertainty.AbstractArray)
        assert np.all(c == d)
        if isinstance(b, kgpy.uncertainty.AbstractArray):
            assert np.all(c.nominal == a.nominal * b.nominal)
            assert np.all(c.distribution == a.distribution * b.distribution)
            assert np.all(d.nominal == b.nominal * a.nominal)
            assert np.all(d.distribution == b.distribution * a.distribution)
        else:
            assert np.all(c.nominal == a.nominal * b)
            assert np.all(c.distribution == a.distribution * b)
            assert np.all(d.nominal == b * a.nominal)
            assert np.all(d.distribution == b * a.distribution)

    @pytest.mark.parametrize(
        argnames='a',
        argvalues=[
            kgpy.uncertainty.Uniform(nominal=1, width=1)
        ],
    )
    @pytest.mark.parametrize(
        argnames='b',
        argvalues=[
            kgpy.uncertainty.Uniform(nominal=2, width=1)
        ],
    )
    def test__array_function__stack(self, a: kgpy.uncertainty.ArrayLike, b: kgpy.uncertainty.ArrayLike):
        result = np.stack([a, b], 'y')
        if not isinstance(a.nominal, kgpy.labeled.AbstractArray):
            a.nominal = kgpy.labeled.Array(a.nominal)
        if not isinstance(a.distribution, kgpy.labeled.AbstractArray):
            a.distribution = kgpy.labeled.Array(a.nominal)
        if not isinstance(b.nominal, kgpy.labeled.AbstractArray):
            b.nominal = kgpy.labeled.Array(b.nominal)
        if not isinstance(b.distribution, kgpy.labeled.AbstractArray):
            b.distribution = kgpy.labeled.Array(b.nominal)
        assert np.all(result.nominal == np.stack([a.nominal, b.nominal], 'y'))
        assert np.all(result.distribution == np.stack([a.distribution, b.distribution], 'y'))
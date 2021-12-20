import pytest
import abc
import numpy as np
import astropy.units as u
import kgpy.labeled
import kgpy.distribution


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
        a = kgpy.distribution.Uniform(nominal=nominal, width=width, num_samples=11)
        assert isinstance(a, kgpy.distribution.Uniform)
        assert isinstance(a.distribution, kgpy.labeled.UniformRandomSpace)
        if not isinstance(nominal, kgpy.labeled.AbstractArray):
            value = kgpy.labeled.Array(nominal)
        assert np.all(a.distribution.min(axis='_distribution') >= nominal - width)


class TestNormal(_TestAbstractArray):

    pass


class TestArray:

    @pytest.mark.parametrize(
        argnames='a,b',
        argvalues=[
            (kgpy.distribution.Uniform(10, width=1), 5),
            (kgpy.distribution.Uniform(10, width=1), kgpy.labeled.LinearSpace(5, 6, 9, axis='b')),
            (kgpy.distribution.Uniform(kgpy.labeled.LinearSpace(10, 11, 7, axis='a'), width=1), 5),
            (kgpy.distribution.Uniform(kgpy.labeled.LinearSpace(10, 11, 7, axis='a'), width=1), kgpy.labeled.LinearSpace(5, 6, 9, axis='b')),
            (kgpy.distribution.Uniform(10 * u.mm, width=1*u.mm), 5 * u.mm),
            (kgpy.distribution.Uniform(10 * u.mm, width=1 * u.mm), kgpy.labeled.LinearSpace(5, 6, 9, axis='b') * u.mm),
            (kgpy.distribution.Uniform(kgpy.labeled.LinearSpace(10, 11, 7, axis='a') * u.mm, width=1 * u.mm), 5 * u.mm),
            (kgpy.distribution.Uniform(kgpy.labeled.LinearSpace(10, 11, 7, axis='a') * u.mm, width=1 * u.mm), kgpy.labeled.LinearSpace(5, 6, 9, axis='b') * u.mm)
        ],
    )
    def test__add__(self, a, b):
        c = a + b
        d = b + a
        b_normalized = b
        if not isinstance(b, kgpy.distribution.AbstractArray):
            b_normalized = kgpy.distribution.Array(b)
        assert isinstance(c, kgpy.distribution.AbstractArray)
        assert isinstance(d, kgpy.distribution.AbstractArray)
        assert np.all(c.nominal == a.nominal + b_normalized.nominal)
        assert np.all(d.nominal == b_normalized.nominal + a.nominal)
        assert np.all(c == d)

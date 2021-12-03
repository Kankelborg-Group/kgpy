import pytest
import abc
import numpy as np
import astropy.units as u
import kgpy.labeled
import kgpy.distribution


class _TestAbstractArray(
    abc.ABC
):

    @property
    @abc.abstractmethod
    def array(self) -> kgpy.distribution.AbstractArray:
        pass

    def test_distribution(self):
        assert isinstance(self.array.distribution, kgpy.labeled.Array)
        assert self.array.distribution.mean() != 0


class TestUniform(_TestAbstractArray):

    @property
    def array(self) -> kgpy.distribution.Uniform:
        return kgpy.distribution.Uniform(
            value=10 * u.mm,
            width=1 * u.mm,
            num_samples=11,
        )

    @pytest.mark.parametrize(
        argnames='value,width',
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
    def test_distribution(self, value: kgpy.labeled.ArrayLike, width: kgpy.labeled.ArrayLike):
        a = kgpy.distribution.Uniform(value=value, width=width, num_samples=11)
        assert isinstance(a, kgpy.distribution.Uniform)
        assert isinstance(a.distribution, kgpy.labeled.UniformRandomSpace)
        if not isinstance(value, kgpy.labeled.AbstractArray):
            value = kgpy.labeled.Array(value)
        assert np.all(a.distribution.min(axis='_distribution') >= value - width)


class TestNormal(_TestAbstractArray):

    @property
    def array(self) -> kgpy.distribution.Normal:
        return kgpy.distribution.Normal(
            value=10 * u.mm,
            width=1 * u.mm,
            num_samples=11,
        )

import abc
import astropy.units as u
import kgpy.distribution


class _TestAbstractArray(
    abc.ABC
):

    @property
    @abc.abstractmethod
    def array(self) -> kgpy.distribution.AbstractArray:
        pass

    def test_distribution(self):
        print(self.array.distribution)


class TestUniform(_TestAbstractArray):

    @property
    def array(self) -> kgpy.distribution.Uniform:
        return kgpy.distribution.Uniform(
            value=10 * u.mm,
            width=1 * u.mm,
            num_samples=11,
        )

import matplotlib.axes
import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vector


@pytest.mark.parametrize(
    argnames='unit',
    argvalues=[1, u.mm]
)
@pytest.mark.parametrize(
    argnames='x',
    argvalues=[
        2,
        kgpy.labeled.LinearSpace(0, 10, num=10, axis='a'),
    ]
)
@pytest.mark.parametrize(
    argnames='x_width',
    argvalues=[None, 0.1],
)
@pytest.mark.parametrize(
    argnames='y',
    argvalues=[
        3,
        kgpy.labeled.LinearSpace(0, 10, num=11, axis='b'),
    ]
)
@pytest.mark.parametrize(
    argnames='y_width',
    argvalues=[None, 0.1],
)
class TestCartesian2D:

    @classmethod
    def factory(
            cls,
            unit: u.Unit,
            x: kgpy.labeled.ArrayLike,
            x_width: kgpy.labeled.ArrayLike,
            y: kgpy.labeled.ArrayLike,
            y_width: kgpy.labeled.ArrayLike,
    ) -> kgpy.vector.Cartesian2D:
        if x_width is not None:
            x = kgpy.uncertainty.Uniform(x, x_width)
        if y_width is not None:
            y = kgpy.uncertainty.Uniform(y, y_width)
        return kgpy.vector.Cartesian2D(x, y) * unit

    def test__mul__(
            self,
            unit: u.Unit,
            x: kgpy.labeled.ArrayLike,
            x_width: kgpy.labeled.ArrayLike,
            y: kgpy.labeled.ArrayLike,
            y_width: kgpy.labeled.ArrayLike,
    ):
        a = self.factory(unit, x, x_width, y, y_width)

        b = a.copy()
        b.x = 2 * a.x
        b.y = -3 * a.y

        c = a * b
        d = b * a
        assert isinstance(c, kgpy.vector.Cartesian2D)
        assert isinstance(d, kgpy.vector.Cartesian2D)
        assert np.all(c == d)
        if not isinstance(b, kgpy.vector.Cartesian2D):
            assert np.all(c.x == a.x * b)
            assert np.all(c.y == a.y * b)
        else:
            assert np.all(c.x == a.x * b.x)
            assert np.all(c.y == a.y * b.y)

    def test_broadcast_to(
            self,
            unit: u.Unit,
            x: kgpy.labeled.ArrayLike,
            x_width: kgpy.labeled.ArrayLike,
            y: kgpy.labeled.ArrayLike,
            y_width: kgpy.labeled.ArrayLike,
    ):
        a = self.factory(unit, x, x_width, y, y_width)
        b = np.broadcast_to(a, a.shape)
        assert b.x.shape == a.shape
        assert b.y.shape == a.shape

    def _ax_factory(self) -> matplotlib.axes.Axes:
        fig, ax = plt.subplots()
        return ax

    def test_plot(
            self,
            unit: u.Unit,
            x: kgpy.labeled.ArrayLike,
            x_width: kgpy.labeled.ArrayLike,
            y: kgpy.labeled.ArrayLike,
            y_width: kgpy.labeled.ArrayLike,
    ):
        a = self.factory(unit, x, x_width, y, y_width)

        ax = self._ax_factory()
        lines = a.plot(ax, axis_plot='a', color=a.y)
        assert lines
        # plt.show()
        plt.close(ax.figure)


class TestCartesian3D(TestCartesian2D):

    @classmethod
    def factory(
            cls,
            unit: u.Unit,
            x: kgpy.labeled.ArrayLike,
            x_width: kgpy.labeled.ArrayLike,
            y: kgpy.labeled.ArrayLike,
            y_width: kgpy.labeled.ArrayLike,
    ) -> kgpy.vector.Cartesian2D:
        result = super().factory(unit, x, x_width, y, y_width)
        result = kgpy.vector.Cartesian3D(
            x=result.x,
            y=result.y,
            z=result.x * result.x + result.y * result.y
        )
        if isinstance(result.z, kgpy.uncertainty.AbstractArray):
            result.z = kgpy.uncertainty.Uniform(result.z.nominal, width=10 * unit * unit)
        return result

    def _ax_factory(self) -> matplotlib.axes.Axes:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        return ax

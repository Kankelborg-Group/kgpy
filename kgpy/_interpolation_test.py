import typing as typ
import numpy as np
import matplotlib.pyplot as plt
from . import interpolation

__all__ = [
    'TestNearestNeighbor'
]


class TestNearestNeighbor:

    @property
    def interpolator_type(self) -> typ.Type[interpolation.Interpolator]:
        return interpolation.NearestNeighbor

    def test__call__1d(self, capsys):

        with capsys.disabled():

            x = np.linspace(0, 2 * np.pi, 10)
            y = np.sin(x)

            interp = self.interpolator_type(
                data=y,
                grid=[x],
            )

            x_new = np.linspace(0, 2 * np.pi, 100)
            y_new = interp([x_new])

            plt.figure()
            plt.scatter(*np.broadcast_arrays(x_new, y_new))
            plt.scatter(*np.broadcast_arrays(x, y))

            plt.figure()
            plt.plot(x, interp([x]) - y)

            plt.show()

            assert np.isclose(interp([x]), y).all()
            # assert (interp([x]) == y).all()

    def test__call__2d(self, capsys):

        with capsys.disabled():

            x = np.linspace(0, 2 * np.pi, 20)[:, np.newaxis]
            y = np.linspace(0, 2 * np.pi, 10)[np.newaxis, :]
            z = np.sin(x) * np.cos(y)

            interp = self.interpolator_type(
                data=z,
                grid=[x, y],
            )

            x_new = np.linspace(0, 2 * np.pi, 2000)[:, np.newaxis]
            y_new = np.linspace(0, 2 * np.pi, 1000)[np.newaxis, :]
            z_new = interp([x_new, y_new])

            x, y, z = np.broadcast_arrays(x, y, z)
            x_new, y_new, z_new = np.broadcast_arrays(x_new, y_new, z_new)

            # plt.figure()
            # plt.scatter(x, y, c=z)
            # plt.colorbar()
            #
            # plt.figure()
            # plt.scatter(x_new, y_new, c=z_new)
            # plt.colorbar()
            #
            # plt.figure()
            # plt.scatter(x, y, c=interp([x, y]) - z)
            # plt.colorbar()
            #
            # plt.show()

            assert np.isclose(interp([x, y]), z).all()


class TestLinear(TestNearestNeighbor):

    @property
    def interpolator_type(self) -> typ.Type[interpolation.Interpolator]:
        return interpolation.Linear

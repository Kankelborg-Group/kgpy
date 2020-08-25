import pytest
import numpy as np
import astropy.units as u
from kgpy import vector
from kgpy.vector import x_hat, y_hat, z_hat
from . import tilt


class TestTiltX:

    @pytest.mark.parametrize(
        argnames='vec, angle, vec_out',
        argvalues=[
            [y_hat, 90 * u.deg, z_hat],
            [y_hat, 180 * u.deg, -y_hat],
            [z_hat, -90 * u.deg, y_hat],
        ],
    )
    def test__call__(self, vec: u.Quantity, angle: u.Quantity, vec_out: u.Quantity):
        assert np.isclose(tilt.TiltX(angle)(vec), vec_out).all()


class TestTiltY:

    @pytest.mark.parametrize(
        argnames='vec, angle, vec_out',
        argvalues=[
            [z_hat, 90 * u.deg, x_hat],
            [z_hat, 180 * u.deg, -z_hat],
            [x_hat, -90 * u.deg, z_hat],
        ],
    )
    def test__call__(self, vec: u.Quantity, angle: u.Quantity, vec_out: u.Quantity):
        assert np.isclose(tilt.TiltY(angle)(vec), vec_out).all()


class TestTiltZ:

    @pytest.mark.parametrize(
        argnames='vec, angle, vec_out',
        argvalues=[
            [x_hat, 90 * u.deg, y_hat],
            [x_hat, 180 * u.deg, -x_hat],
            [y_hat, -90 * u.deg, x_hat],
        ],
    )
    def test__call__(self, vec: u.Quantity, angle: u.Quantity, vec_out: u.Quantity):
        assert np.isclose(tilt.TiltZ(angle)(vec), vec_out).all()

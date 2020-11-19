import pytest
import numpy as np
import astropy.units as u
from kgpy import vector
from kgpy.vector import x_hat, y_hat, z_hat
from . import rigid


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
        assert np.isclose(rigid.TiltX(angle)(vec), vec_out).all()


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
        assert np.isclose(rigid.TiltY(angle)(vec), vec_out).all()


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
        assert np.isclose(rigid.TiltZ(angle)(vec), vec_out).all()


class TestTransformList:

    def test_rot90(self):
        transform = rigid.TransformList([rigid.TiltZ(90 * u .deg)])
        a = vector.x_hat
        b = transform(a)
        c = transform.inverse(b)
        assert np.isclose(b, vector.y_hat).all()
        assert np.isclose(c, a).all()

    def test_polar(self):
        transform = rigid.TransformList([
            rigid.TiltZ(90 * u.deg),
            rigid.Translate([1, 0, 0] * u.dimensionless_unscaled),
        ])
        a = vector.from_components()
        b = transform(a)
        c = transform.inverse(b)
        assert np.isclose(b, vector.y_hat).all()
        assert np.isclose(c, a).all()

    def test_spherical(self):
        transform = rigid.TransformList([
            rigid.TiltZ(90 * u.deg),
            rigid.TiltY(90 * u.deg),
            # Translate([0, 0, 1] * u.dimensionless_unscaled)
        ])
        a = vector.from_components(z=1)
        b = transform(a)
        c = transform.inverse(b)
        assert np.isclose(b, vector.y_hat).all()
        assert np.isclose(c, a).all()

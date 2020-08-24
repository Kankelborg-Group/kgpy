import numpy as np
import astropy.units as u
from kgpy import vector
from . import TransformList, Translate, TiltX, TiltY, TiltZ


class TestTransformList:

    def test_rot90(self):
        transform = TransformList([TiltZ(90 * u .deg)])
        a = vector.x_hat
        b = transform(a)
        c = transform.inverse(b)
        assert np.isclose(b, vector.y_hat).all()
        assert np.isclose(c, a).all()

    def test_polar(self):
        transform = TransformList([
            TiltZ(90 * u.deg),
            Translate([1, 0, 0] * u.dimensionless_unscaled),
        ])
        a = vector.from_components()
        b = transform(a)
        c = transform.inverse(b)
        assert np.isclose(b, vector.y_hat).all()
        assert np.isclose(c, a).all()

    def test_spherical(self):
        transform = TransformList([
            TiltZ(90 * u.deg),
            TiltY(90 * u.deg),
            # Translate([0, 0, 1] * u.dimensionless_unscaled)
        ])
        a = vector.from_components(z=1)
        b = transform(a)
        c = transform.inverse(b)
        print(transform.inverse)
        print(list(transform.inverse.transforms))
        assert np.isclose(b, vector.y_hat).all()
        assert np.isclose(c, a).all()

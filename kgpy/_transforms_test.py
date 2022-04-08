import pytest
import astropy.units as u
import kgpy.vectors
import kgpy.transforms


class TestTransformList:

    @pytest.mark.parametrize(
        argnames='a',
        argvalues=[
            kgpy.transforms.TransformList([
                kgpy.transforms.Translation(kgpy.vectors.Cartesian3D(x=2) * u.m),
                kgpy.transforms.RotationZ(90 * u.deg),
                kgpy.transforms.Translation(kgpy.vectors.Cartesian3D(x=2) * u.m),
                kgpy.transforms.RotationY(90 * u.deg),
                kgpy.transforms.Translation(kgpy.vectors.Cartesian3D(x=2) * u.m),
            ])
        ]
    )
    def test__call__(self, a: kgpy.transforms.TransformList):
        x = kgpy.vectors.Cartesian3D() * u.m
        b = a(x)
        c = x
        for transform in a.transforms:
            c = transform(c)
        assert b == c

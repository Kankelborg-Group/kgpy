import pytest
import astropy.units as u
import matplotlib.pyplot as plt
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vector
import kgpy.transforms
import kgpy.optics.surface.aperture
import kgpy.optics.surface.sag


class TestRectangular:

    @pytest.mark.parametrize(
        argnames='half_width',
        argvalues=[
            kgpy.vector.Cartesian2D(20, 10) * u.mm
        ],
    )
    def test_vertices(self, half_width: kgpy.vector.Cartesian2D):
        aper = kgpy.optics.surface.aperture.Rectangular(half_width=half_width)
        assert isinstance(aper.vertices, kgpy.vector.Cartesian3D)

    @pytest.mark.parametrize(
        argnames='half_width',
        argvalues=[
            kgpy.vector.Cartesian2D(20, 10) * u.mm,
            kgpy.vector.Cartesian2D(kgpy.labeled.LinearSpace(20, 30, num=3, axis='chan'), 10) * u.mm,
        ],
    )
    @pytest.mark.parametrize(
        argnames='decenter_x',
        argvalues=[
            5 * u.mm,
            kgpy.labeled.LinearSpace(-100 * u.mm, 100 * u.mm, num=3, axis='chan'),
            kgpy.uncertainty.Uniform(5 * u.mm, width=1 * u.mm)
        ],
    )
    def test_plot(self, decenter_x: kgpy.uncertainty.ArrayLike, half_width: kgpy.vector.Cartesian2D):
        fig, ax = plt.subplots()
        aperture = kgpy.optics.surface.aperture.Rectangular(
            transform=kgpy.transforms.Translation(kgpy.vector.Cartesian3D(decenter_x, 0 * u.mm, 0 * u.mm)),
            # transform=kgpy.transforms.TransformList([
            #     kgpy.transforms.Translation(kgpy.vector.Cartesian3D(decenter_x, 0 * u.mm, 0 * u.mm)),
            # ]),
            half_width=half_width
        )
        aperture.plot(ax)
        plt.show()

    @pytest.mark.parametrize(
        argnames='decenter_x,facecolor',
        argvalues=[
            (0 * u.mm, 'blue'),
            (kgpy.labeled.LinearSpace(-50 * u.mm, 50 * u.mm, num=3, axis='chan'), kgpy.labeled.Array(['blue', 'green', 'red'], axes=['chan'])),
        ]
    )
    def test_plot_3d(self, decenter_x: kgpy.uncertainty.ArrayLike, facecolor: str):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        sag = kgpy.optics.surface.sag.Standard(radius=1000 * u.mm)
        aperture = kgpy.optics.surface.aperture.Rectangular(
            transform=kgpy.transforms.Translation(kgpy.vector.Cartesian3D(decenter_x, 0 * u.mm, 0 * u.mm)),
            # transform=kgpy.transforms.TransformList([
            #     kgpy.transforms.Translation(kgpy.vector.Cartesian3D(decenter_x, 0 * u.mm, 0 * u.mm)),
            # ]),
            half_width=kgpy.vector.Cartesian2D(10 * u.mm, 20 * u.mm),
        )
        assert aperture.plot(
            ax=ax,
            component_z='z',
            sag=sag,
            facecolor=facecolor,
        )
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)
        plt.show()


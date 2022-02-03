import pytest
import astropy.units as u
import matplotlib.pyplot as plt
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vector
import kgpy.transforms
import kgpy.optics.surface.aperture
import kgpy.optics.surface.sag
import kgpy.optics.surface.material


class TestMirror:

    @pytest.mark.parametrize(
        argnames='decenter_x, color',
        argvalues=[
            (0, 'black'),
            (kgpy.labeled.LinearSpace(-500, 500, 3, axis='chan'), kgpy.labeled.Array(['black', 'blue', 'green'], axes=['chan']))
        ]
    )
    def test_plot(self, decenter_x, color):
        fig, ax = plt.subplots()
        sag = kgpy.optics.surface.sag.Standard(radius=10000 * u.mm)
        aperture = kgpy.optics.surface.aperture.RegularPolygon(
            transform=kgpy.transforms.Translation(kgpy.vector.Cartesian3D(x=decenter_x) * u.mm),
            radius=100 * u.mm,
            num_sides=8,
        )
        material = kgpy.optics.surface.material.Mirror(thickness=10 * u.mm)
        material.plot(
            ax=ax,
            sag=sag,
            aperture=aperture,
            component_x='z',
            component_y='x',
            color=color,
        )
        plt.show()

    def test_plot_3d(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        sag = kgpy.optics.surface.sag.Standard(radius=10000 * u.mm)
        aperture = kgpy.optics.surface.aperture.RegularPolygon(
            transform=kgpy.transforms.Translation(kgpy.vector.Cartesian3D(x=0) * u.mm),
            radius=100 * u.mm,
            num_sides=8,
        )
        material = kgpy.optics.surface.material.Mirror(thickness=20 * u.mm)
        material.plot(
            ax=ax,
            sag=sag,
            aperture=aperture,
            # component_x='z',
            # component_y='x',
            component_z='z',
            facecolor='white',
            edgecolor='black',
            # color=,
        )
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)
        plt.show()



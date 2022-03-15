import matplotlib.colors
import pytest
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.transforms
import kgpy.optics


class TestSystem:

    def test_raytrace(self, capsys):
        with capsys.disabled():

            focal_length = 1000 * u.mm

            system = kgpy.optics.systems.System(
                object_surface=kgpy.optics.surfaces.Surface(
                    name='sky',
                    transform=kgpy.transforms.TransformList([kgpy.transforms.Translation(kgpy.vectors.Cartesian3D(z=-100) * u.mm)]),
                ),
                surfaces=kgpy.optics.surfaces.SurfaceList([
                    kgpy.optics.surfaces.Surface(
                        name='primary',
                        transform=kgpy.transforms.TransformList([
                            # kgpy.transforms.RotationY(30 * u.deg),
                            kgpy.transforms.Translation(kgpy.vectors.Cartesian3D(
                                # x=kgpy.uncertainty.Uniform(0 * u.mm, width=10 * u.mm),
                                # y=kgpy.uncertainty.Uniform(0 * u.mm, width=10 * u.mm),
                                x=0 * u.mm,
                                y=0 * u.mm,
                                z=focal_length,
                            )),
                            # kgpy.transforms.RotationZ(45 * u.deg)
                            # kgpy.transforms.RotationY(-15 * u.deg),
                            # kgpy.transforms.RotationY(180 * u.deg),
                        ]),
                        is_pupil_stop=True,
                        sag=kgpy.optics.surfaces.sags.Standard(radius=-2 * focal_length),
                        material=kgpy.optics.surfaces.materials.Mirror(thickness=-10 * u.mm),
                        aperture=kgpy.optics.surfaces.apertures.Rectangular(half_width=kgpy.vectors.Cartesian2D() + 50 * u.mm)
                    ),
                    kgpy.optics.surfaces.Surface(
                        name='image',
                        is_field_stop=True,
                        aperture=kgpy.optics.surfaces.apertures.Rectangular(half_width=kgpy.vectors.Cartesian2D() + 5 * u.mm),
                    )
                ]),
                object_grid_normalized=kgpy.optics.vectors.ObjectVector(
                    field=kgpy.vectors.Cartesian2D(
                        x=kgpy.labeled.LinearSpace(-1, 1, 5, axis='field.x'),
                        y=kgpy.labeled.LinearSpace(-1, 1, 5, axis='field.y'),
                    ),
                    pupil=kgpy.vectors.Cartesian2D(
                        x=kgpy.labeled.LinearSpace(-1, 1, 101, axis='pupil.x'),
                        y=kgpy.labeled.LinearSpace(-1, 1, 101, axis='pupil.y'),
                    ),
                    wavelength=500 * u.nm,
                )
            )
            # rays_input= system._calc_rays_input_stops_only(object_grid_normalized)
            # rays_input = system._calc_rays_input_stops(system.object_grid_normalized)
            # rays_input = system._calc_rays_input(system.object_grid_normalized)
            # rays = kgpy.optics.rays.RayFunctionList([rays_input_stops_only]) + rays
            # print(rays.intercepts.shape)
            # print(rays_input.output)

            # rays = system.surfaces_all.flat_global.raytrace(rays_input)

            rays = system.rays_output


            # fig, ax = plt.subplots(
            #     # subplot_kw=dict(projection='3d'),
            # )
            # system.plot(
            #     ax=ax,
            #     component_x='z',
            #     component_y='x',
            #     component_z='y',
            #     # plot_rays=False,
            #     # plot_vignetted=True,
            #     plot_annotations=False,
            #     # plot_baffles=False,
            #     # plot_breadboard=False,
            # )

            fig_psf, ax_psf = plt.subplots(
                nrows=rays.input.field.x.num,
                ncols=rays.input.field.y.num,
                sharex=True,
                sharey=True,
                squeeze=False,
                constrained_layout=True,
            )
            point_spread = rays.point_spread()
            point_spread.function.pcolormesh(
                axs=ax_psf,
                input_component_x='position.x',
                input_component_y='position.y',
                input_component_row='field.y',
                input_component_column='field.x',
                # index={'field.x': 2, 'field.y': 2},
                norm=matplotlib.colors.PowerNorm(
                    gamma=1 / 2,
                    vmin=point_spread.function.output.min().array,
                    vmax=point_spread.function.output.max().array,
                ),
            )

            # plt.show()
            plt.close(fig_psf)


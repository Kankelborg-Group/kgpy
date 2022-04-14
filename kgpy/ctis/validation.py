import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.optics
import kgpy.function

__all__ = []


def simple_emission_line() -> kgpy.function.Array[kgpy.optics.vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:

    inp = kgpy.optics.vectors.SpectralFieldVector(
        field=kgpy.vectors.Cartesian2D(
            x=kgpy.labeled.LinearSpace(0, 25, num=501, axis='field_x'),
            y=kgpy.labeled.LinearSpace(0, 20, num=503, axis='field_y'),
        ),
        wavelength=kgpy.labeled.Array([304] * u.AA, axes=['wavelength']),
        velocity_los=kgpy.labeled.LinearSpace(-100 * u.km / u.s, 100 * u.km / u.s, num=11, axis='velocity_los')
    )

    points = kgpy.vectors.Cartesian2D.stratified_random_space(
        start=kgpy.vectors.Cartesian2D(inp.field.x.start, inp.field.y.start),
        stop=kgpy.vectors.Cartesian2D(inp.field.x.stop, inp.field.y.stop),
        num=kgpy.vectors.Cartesian2D(11, 13),
        axis=kgpy.vectors.Cartesian2D('field_x', 'field_y'),
    )
    # points = points.broadcasted
    print(points)
    # points = points.broadcasted.combine_axes(axes=['points_x', 'points_y'], axis_new='points')

    # points = kgpy.vectors.Cartesian2D(
    #     x=kgpy.labeled.LinearSpace(0, 25, num=5, axis='field_x'),
    #     y=kgpy.labeled.LinearSpace(0, 20, num=5, axis='field_y'),
    # )
    # points = points.broadcasted

    # points = inp.field.broadcasted

    intensity = kgpy.function.Array(
        input=points,
        output=kgpy.labeled.Array.ones(points.shape),
    )

    # intensity.output = np.cos(points.x) * np.cos(points.y)

    for index in intensity.output.ndindex():
        if sum(index.values()) % 2 != 0:
            intensity.output[index] = -intensity.output[index]

    fig, axs = plt.subplots(squeeze=False)
    intensity.pcolormesh(
        axs=axs,
        input_component_x='x',
        input_component_y='y',
        vmin=-1,
        vmax=1,
        # cmap='gray',
    )
    # points.scatter(
    #     ax=axs[0, 0],
    #     # axis_plot='field_x',
    #     axis_plot=None,
    #     color=intensity.output,
    # )
    axs[0, 0].set_xlim((-5, 30))
    axs[0, 0].set_ylim((-5, 30))

    fig2, axs2 = plt.subplots(squeeze=False)

    intensity2 = intensity.interp_barycentric_linear(input_new=inp.field)

    print('intensity2.input', intensity2.input.shape)
    print('intensity2.output', intensity2.output.shape)

    intensity2.pcolormesh(
        axs=axs2,
        input_component_x='x',
        input_component_y='y',
        vmin=-1,
        vmax=1,
        # cmap='gray',
    )
    axs2[0, 0].set_xlim((-5, 30))
    axs2[0, 0].set_ylim((-5, 30))

    print(points.centers)

    # points.centers.scatter(
    #     ax=axs2[0, 0],
    #     # axis_plot='field_x',
    #     axis_plot=None,
    #     color=intensity.output,
    #     cmap='gray',
    # )

    # fig3, ax3 = plt.subplots()
    # inp.field.scatter(
    #     ax3,
    #     axis_plot='field_x',
    #     cmap='viridis'
    #     # color='tab:blue',
    #     # color=intensity2.output,
    # )
    # points.scatter(
    #     ax3,
    #     axis_plot='field_x',
    #     cmap='viridis',
    #     # color='tab:orange',
    # )




    plt.show()





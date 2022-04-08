import astropy.units as u
import matplotlib.pyplot as plt
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.optics
import kgpy.function

__all__ = []


def simple_emission_line() -> kgpy.function.Array[kgpy.optics.vectors.FieldVector, kgpy.uncertainty.ArrayLike]:

    inp = kgpy.optics.vectors.FieldVector(
        field=kgpy.vectors.Cartesian2D(
            x=kgpy.labeled.LinearSpace(-10 * u.arcmin, 10 * u.arcmin, num=101, axis='field_x'),
            y=kgpy.labeled.LinearSpace(-10 * u.arcmin, 10 * u.arcmin, num=101, axis='field_y'),
        ),
        wavelength=kgpy.labeled.Array([304] * u.AA, axes=['wavelength']),
        velocity_los=kgpy.labeled.LinearSpace(-100 * u.km / u.s, 100 * u.km / u.s, num=11, axis='velocity_los')
    )

    points = kgpy.vectors.Cartesian2D.stratified_random_space(
        start=kgpy.vectors.Cartesian2D(inp.field.x.start, inp.field.y.start),
        stop=kgpy.vectors.Cartesian2D(inp.field.x.stop, inp.field.y.stop),
        num=kgpy.vectors.Cartesian2D(11, 11),
        axis=kgpy.vectors.Cartesian2D('field_x', 'field_y'),
    )
    print(points)
    # points = points.broadcasted.combine_axes(axes=['points_x', 'points_y'], axis_new='points')

    points = kgpy.vectors.Cartesian2D(
        x=kgpy.labeled.LinearSpace(-10 * u.arcmin, 10 * u.arcmin, num=11, axis='field_x'),
        y=kgpy.labeled.LinearSpace(-10 * u.arcmin, 10 * u.arcmin, num=11, axis='field_y'),
    )
    points = points.broadcasted

    intensity = kgpy.function.Array(
        input=points,
        output=kgpy.labeled.Array.ones(points.shape),
    )

    for index in intensity.output.ndindex():
        if sum(index.values()) % 2 != 0:
            intensity.output[index] = -intensity.output[index]

    fig, axs = plt.subplots(squeeze=False)
    intensity.pcolormesh(
        axs=axs,
        input_component_x='x',
        input_component_y='y',
    )

    intensity = intensity.interp_barycentric_linear(input_new=inp.field)

    fig2, axs2 = plt.subplots(squeeze=False)
    intensity.pcolormesh(
        axs=axs2,
        input_component_x='x',
        input_component_y='y',
    )
    axs2[0, 0].set_xlim(axs[0, 0].get_xlim())
    axs2[0, 0].set_ylim(axs[0, 0].get_ylim())
    # points.scatter(
    #     ax=ax,
    #     axis_plot='points',
    #     # color=intensity.output,
    # )
    plt.show()





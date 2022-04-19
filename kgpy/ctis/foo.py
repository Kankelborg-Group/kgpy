import abc
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants
# %%
import kgpy.labeled
import kgpy.function
import kgpy.vectors

# %%
x = kgpy.labeled.LinearSpace(0, 1, num=11, axis='x')
y = kgpy.labeled.LinearSpace(0, 1, num=11, axis='y')
# %%
intensity = x + y + 2 * x * y
# %%
a = kgpy.function.PolynomialArray(
    input=kgpy.vectors.Cartesian2D(x, y),
    output=kgpy.vectors.Cartesian2D(intensity, -intensity),
    degree=2,
)
# %%
a.coefficients
# %%
import astropy.units as u
import kgpy.ctis.instruments
import kgpy.optics.aberrations

# %%
instrument = kgpy.ctis.instruments.AberrationInstrument(
    aberration=kgpy.optics.aberrations.Aberration(
        distortion=kgpy.optics.aberrations.Distortion(kgpy.function.Polynomial(
            # input=kgpy.optics.vectors.SpectralFieldVector(
            #     field_x=kgpy.labeled.LinearSpace(-0.25 * u.deg, 0.25 * u.deg, num=100, axis='field_x'),
            #     field_y=kgpy.labeled.LinearSpace(-0.25 * u.deg, 0.25 * u.deg, num=100, axis='field_x'),
            #     wavelength=kgpy.labeled.LinearSpace(550 * u.AA, 650 * u.AA, num=10, axis='wavelength'),
            # ),
            input=None,
            coefficients=kgpy.vectors.CartesianND({
                '': kgpy.optics.vectors.SpectralPositionVector(
                    position_x=-574 * u.pix,
                    position_y=10 * u.pix,
                    wavelength=0 * u.AA,
                ),
                'field_x': kgpy.optics.vectors.SpectralPositionVector(
                    position_x=kgpy.labeled.Array([1, 0] * u.pix / u.arcsec, axes=['channel']),
                    position_y=kgpy.labeled.Array([0, 1] * u.pix / u.arcsec, axes=['channel']),
                    wavelength=0 * u.AA / u.arcsec,
                ),
                'field_y': kgpy.optics.vectors.SpectralPositionVector(
                    position_x=kgpy.labeled.Array([0, 1] * u.pix / u.arcsec, axes=['channel']),
                    position_y=kgpy.labeled.Array([1, 0] * u.pix / u.arcsec, axes=['channel']),
                    wavelength=0 * u.AA / u.arcsec,
                ),
                'wavelength': kgpy.optics.vectors.SpectralPositionVector(
                    position_x=1 * u.pix / u.AA,
                    position_y=0 * u.pix / u.AA,
                    wavelength=1,
                ),
            })
        )
        ))
)
# %% md

# %%
wvs = kgpy.labeled.Array([584, 602, 630] * u.AA, axes=['wavelength'])
inputs = kgpy.optics.vectors.SpectralFieldVector(
    field_x=kgpy.labeled.LinearSpace(-10 * u.arcsec, 10 * u.arcsec, num=11, axis='field_x'),
    field_y=kgpy.labeled.LinearSpace(-10 * u.arcsec, 10 * u.arcsec, num=11, axis='field_y'),
    wavelength=wvs,
    # wavelength=kgpy.optics.vectors.DopplerVector(
    #     wavelength_rest=wavelength_rest,
    #     # velocity_los=kgpy.labeled.LinearSpace(-10 * u.km / u.s, 10 * u.km / u.s, num=3, axis='velocity_los'),
    #     velocity_los=kgpy.labeled.LinearSpace(-.300 * u.AA, .3 * u.AA, num=3, axis='velocity_los') / wavelength_rest * astropy.constants.c,
    # ).wavelength
)

output = np.exp(-np.square(inputs.field_x / (10 * u.arcsec))) * np.exp(-np.square(inputs.field_y / (30 * u.arcsec)))

scene = kgpy.function.Array(
    input=inputs,
    output=output,
)
# %%
fig, ax = plt.subplots(
    nrows=inputs.wavelength.shape['wavelength'],
    # ncols=inputs.velocity_los.num,
    squeeze=False,
    sharex=True,
    sharey=True,
)
scene.pcolormesh(
    axs=ax,
    input_component_x='field_x',
    input_component_y='field_y',
    input_component_row='wavelength',
    # input_component_column='velocity_los',
)
# %%
images = instrument(scene)
# %%
fig_images, ax_images = plt.subplots(
    # figsize=(4, 12),
    nrows=inputs.wavelength.shape['wavelength'],
    # ncols=inputs.velocity_los.num,
    squeeze=False,
    sharex=True,
    sharey=True,
    subplot_kw=dict(aspect='equal'),
)
images.pcolormesh(
    axs=ax_images,
    input_component_x='position_x',
    input_component_y='position_y',
    input_component_row='wavelength',
    index=dict(channel=0)
    # input_component_column='velocity_los',
)
# %%
fig_images, ax_images = plt.subplots(
    # figsize=(4,12),
    nrows=inputs.wavelength.shape['wavelength'],
    # ncols=inputs.velocity_los.num,
    squeeze=False,
    sharex=True,
    sharey=True,
    subplot_kw=dict(aspect='equal'),
)
images.pcolormesh(
    axs=ax_images,
    input_component_x='position_x',
    input_component_y='position_y',
    input_component_row='wavelength',
    index=dict(channel=1)
    # input_component_column='velocity_los',
)
# %%
# fig2, ax2 = plt.subplots()
# ax2.pcolormesh(
#     images.input_broadcasted.position_x.array,
#     images.input_broadcasted.position_y.array,
#     images.output.array,
# )
# %%
plt.show()

input_coords = kgpy.optics.vectors.SpectralPositionVector(
    position_x=kgpy.labeled.LinearSpace(1 * u.pix, 55 * u.pix, num=21, axis='position_x'),
    position_y=kgpy.labeled.LinearSpace(1 * u.pix, 19 * u.pix, num=21, axis='position_y'),
    wavelength=wvs)

import warnings

warnings.filterwarnings("error")

test = images.interp_barycentric_linear(
    input_new=input_coords,
    axis=('wavelength', 'field_x', 'field_y')
)

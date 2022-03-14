import typing as typ
import dataclasses
import time
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import astropy.units as u
import astropy.constants
from kgpy import mixin, obs, plot
from . import Inversion

__all__ = ['CNN']


@dataclasses.dataclass
class CNN(Inversion, mixin.Pickleable):
    model_forward: typ.Callable
    model_inverse: keras.Sequential

    @classmethod
    def train(
            cls,
            model_forward: typ.Callable,
            cube_training: obs.spectral.Cube,
            cube_validation: obs.spectral.Cube,
    ) -> 'CNN':

        cube_training_median = np.mean(cube_training.intensity, axis=(~2, ~1, ~0))[..., None, None, None]
        cube_validation_median = np.mean(cube_validation.intensity, axis=(~2, ~1, ~0))[..., None, None, None]

        cube_training.intensity /= cube_training_median
        cube_validation.intensity /= cube_validation_median

        wcs_sample = cube_training.wcs.flat[0]
        iris_wavelength_nominal = cube_training.channel[0]
        iris_fov_min = wcs_sample.pixel_to_world(0, 0, 0)
        iris_fov_max = wcs_sample.pixel_to_world(
            cube_training.num_wavelength, cube_validation.num_x, cube_training.num_y)

        doppler_min = astropy.constants.c * (iris_fov_min[0] - iris_wavelength_nominal) / iris_wavelength_nominal
        doppler_max = astropy.constants.c * (iris_fov_max[0] - iris_wavelength_nominal) / iris_wavelength_nominal

        print(doppler_min, doppler_max)

        esis_wavelength_nominal = model_forward.wavelengths[~0]
        esis_wavelength_min = doppler_min * esis_wavelength_nominal / astropy.constants.c + esis_wavelength_nominal
        esis_wavelength_max = doppler_max * esis_wavelength_nominal / astropy.constants.c + esis_wavelength_nominal

        wavelength = np.linspace(esis_wavelength_min, esis_wavelength_max, cube_training.num_wavelength)
        spatial_domain_input = u.Quantity([u.Quantity(iris_fov_min[1:]), u.Quantity(iris_fov_max[1:])])
        # spatial_domain_input /= 3

        spatial_domain_output = [[1024, 0], model_forward.detector.num_pixels] * u.pix

        def forward(cube: obs.spectral.Cube):
            spatial_samples = cube.shape[~2:~0]
            projections = model_forward(
                data=np.moveaxis(cube.intensity, cube.axis.w, ~2),
                wavelength=wavelength,
                spatial_domain_input=spatial_domain_input,
                spatial_domain_output=spatial_domain_output,
                spatial_samples_output=spatial_samples
            )
            projections = model_forward(
                data=np.broadcast_to(projections.sum(~2, keepdims=True), projections.shape, subok=True),
                wavelength=wavelength,
                spatial_domain_input=spatial_domain_output,
                spatial_domain_output=spatial_domain_input,
                spatial_samples_output=spatial_samples,
                inverse=True,
            )
            projections = np.moveaxis(projections, ~2, cube_training.axis.w)
            return projections

        projections_training = forward(cube_training)
        projections_validation = forward(cube_validation)

        wavl_trim_sh = cube_training.num_wavelength // 2 // 2 * 2 * 2

        # hs1 = plot.HypercubeSlicer(cube_validation.intensity[..., :wavl_trim_sh].sum(1).value, wcs_list=cube_validation.wcs[:, 0], width_ratios=(5, 1), height_ratios=(5, 1))
        # hs2 = plot.HypercubeSlicer(projections_validation[..., :wavl_trim_sh].sum(1).value, wcs_list=cube_validation.wcs[:, 0], width_ratios=(5, 1), height_ratios=(5, 1))
        # plt.show()

        input_shape = (projections_training.shape[1], None, None, None)
        print(input_shape)

        net = cls.model_inverse_initial(input_shape)

        print(projections_training.shape)
        print(cube_training.intensity.shape)

        tensorboard_dir = pathlib.Path(__file__).parent / 'logs'

        history = net.fit(
            x=np.nan_to_num(projections_training[..., :wavl_trim_sh]),
            y=np.nan_to_num(cube_training.intensity[..., :wavl_trim_sh]),
            batch_size=1,
            epochs=200,
            verbose=2,
            callbacks=[keras.callbacks.TensorBoard(
                log_dir=tensorboard_dir / time.strftime("%Y%m%d-%H%M%S"),
                histogram_freq=0,
                write_graph=False,
                write_images=False,
            )],
            validation_data=(
                np.nan_to_num(projections_validation[..., :wavl_trim_sh]),
                np.nan_to_num(cube_validation.intensity[..., :wavl_trim_sh]),
            ),
        )

        predictions_validation = net.predict(np.nan_to_num(projections_validation[..., :wavl_trim_sh]), batch_size=1)

        hs1 = plot.HypercubeSlicer(cube_validation.intensity[..., :wavl_trim_sh].sum(1).value, wcs_list=cube_validation.wcs[:, 0], width_ratios=(5, 1), height_ratios=(5, 1))
        hs2 = plot.HypercubeSlicer(projections_validation[..., :wavl_trim_sh].sum(1).value, wcs_list=cube_validation.wcs[:, 0], width_ratios=(5, 1), height_ratios=(5, 1))
        hs3 = plot.HypercubeSlicer(predictions_validation.sum(1), wcs_list=cube_validation.wcs[:, 0], width_ratios=(5, 1), height_ratios=(5, 1))
        hs4 = plot.HypercubeSlicer((cube_validation.intensity[..., :wavl_trim_sh].value - predictions_validation).sum(1), wcs_list=cube_validation.wcs[:, 0], width_ratios=(5, 1), height_ratios=(5, 1))

        print(history.history)
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()

        return cls(
            model_forward=model_forward,
            model_inverse=net,
        )

    def __call__(self, projections: obs.Image):
        pass

    def to_pickle(self, path: typ.Optional[pathlib.Path] = None):
        self.model_inverse.save(path.parent / (str(path.stem) + '.h5'))
        self.model_inverse = None
        self.model_forward.update()
        super().to_pickle(path)

    @classmethod
    def from_pickle(cls, path: typ.Optional[pathlib.Path] = None):
        self = super().from_pickle(path)
        self.model_inverse = keras.models.load_model(path.parent / (str(path.stem) + '.h5'))
        return self

    @staticmethod
    def model_inverse_initial(
            input_shape: typ.Tuple[typ.Optional[int], typ.Optional[int], typ.Optional[int], typ.Optional[int]],
            n_filters: int = 32,
            kernel_size: int = 7,
            growth_factor: int = 2,
            alpha: float = 0.1,
            dropout_rate: float = 0.01,
    ) -> keras.Sequential:
        layers = [
            keras.layers.Conv3D(
                filters=n_filters,
                kernel_size=kernel_size,
                padding='same',
                data_format='channels_first',
                input_shape=input_shape,
            ),
            keras.layers.LeakyReLU(
                alpha=alpha,
            ),
            # keras.layers.AveragePooling3D(
            #     padding='same',
            #     data_format='channels_first',
            # ),
            # keras.layers.Dropout(
            #     rate=dropout_rate
            # ),
            keras.layers.Conv3D(
                filters=n_filters * growth_factor,
                kernel_size=kernel_size,
                padding='same',
                data_format='channels_first',
            ),
            keras.layers.LeakyReLU(
                alpha=alpha,
            ),
            # keras.layers.AveragePooling3D(
            #     padding='same',
            #     data_format='channels_first',
            # ),
            # keras.layers.Dropout(
            #     rate=dropout_rate
            # ),
            # keras.layers.Conv3D(
            #     filters=n_filters * growth_factor * growth_factor,
            #     kernel_size=kernel_size,
            #     padding='same',
            #     data_format='channels_first',
            # ),
            # keras.layers.LeakyReLU(
            #     alpha=alpha,
            # ),
            # keras.layers.AveragePooling3D(
            #     pool_size=5,
            #     padding='same',
            #     data_format='channels_first',
            # ),
            # keras.layers.Dropout(
            #     rate=dropout_rate
            # ),
            # keras.layers.Conv3DTranspose(
            #     filters=n_filters * growth_factor,
            #     kernel_size=kernel_size,
            #     padding='same',
            #     data_format='channels_first',
            # ),
            # keras.layers.LeakyReLU(
            #     alpha=alpha,
            # ),
            # keras.layers.UpSampling3D(
            #     size=5,
            #     data_format='channels_first',
            # ),
            # keras.layers.Dropout(
            #     rate=dropout_rate,
            # ),
            keras.layers.Conv3DTranspose(
                filters=n_filters,
                kernel_size=kernel_size,
                padding='same',
                data_format='channels_first',
            ),
            keras.layers.LeakyReLU(
                alpha=alpha,
            ),
            # keras.layers.UpSampling3D(
            #     data_format='channels_first',
            # ),
            # keras.layers.Dropout(
            #     rate=dropout_rate,
            # ),
            keras.layers.Conv3DTranspose(
                filters=n_filters // growth_factor,
                kernel_size=kernel_size,
                padding='same',
                data_format='channels_first',
            ),
            keras.layers.LeakyReLU(
                alpha=alpha,
            ),
            # keras.layers.UpSampling3D(
            #     data_format='channels_first',
            # ),
            # keras.layers.Dropout(
            #     rate=dropout_rate,
            # ),
            keras.layers.Conv3DTranspose(
                filters=1,
                kernel_size=kernel_size,
                padding='same',
                data_format='channels_first',
            )
        ]

        net = keras.Sequential(layers=layers)

        net.compile(
            optimizer=keras.optimizers.Nadam(lr=1e-6),
            loss='mse',
        )

        return net




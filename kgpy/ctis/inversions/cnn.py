from __future__ import annotations
from typing import Type, TypeVar, Sequence
import dataclasses
import pathlib
import time
import numpy as np
import scipy.ndimage
from tensorflow import keras
import tensorflow_addons as tfa
import kgpy.mixin
import kgpy.labeled
import kgpy.vectors
import kgpy.function
import kgpy.optics
import kgpy.solar
from .. import instruments
from . import abstractions


__all__ = [
    'TrainingHistory',
    'Inversion',
]

InversionT = TypeVar('InversionT', bound='Inversion')


@dataclasses.dataclass
class TrainingHistory:
    loss_training: kgpy.labeled.Array
    loss_validation: kgpy.labeled.Array


@dataclasses.dataclass
class Inversion(
    kgpy.mixin.Pickleable,
    abstractions.AbstractInversion,
):

    instrument_inverse: keras.Sequential
    history: TrainingHistory
    average_input: kgpy.labeled.AbstractArray
    average_output: kgpy.labeled.AbstractArray
    width_input: kgpy.labeled.AbstractArray
    width_output: kgpy.labeled.AbstractArray
    num_divisions_fov: kgpy.vectors.Cartesian2D

    @classmethod
    def _output_divided(cls, output: kgpy.labeled.Array, num_divisions_fov: kgpy.vectors.Cartesian2D):
        shape = output.shape
        shape_divided = dict()
        for axis in shape:
            if axis == 'helioprojective_x':
                shape_divided['division_x'] = num_divisions_fov.x
                shape_divided['helioprojective_x'] = shape['helioprojective_x'] // num_divisions_fov.x

            elif axis == 'helioprojective_y':
                shape_divided['division_y'] = num_divisions_fov.y
                shape_divided['helioprojective_y'] = shape['helioprojective_y'] // num_divisions_fov.y

            else:
                shape_divided[axis] = shape[axis]

        result = output.reshape(shape_divided)
        result = result.change_axis_index('division_x', 1)
        result = result.change_axis_index('division_y', 2)

        shape_final = dict()
        for axis in shape:
            if axis == 'time':
                shape_final['time'] = result.shape['time'] * result.shape['division_x'] * result.shape['division_y']

            else:
                shape_final[axis] = result.shape[axis]

        result = result.reshape(shape_final)

        return result

    @classmethod
    def train(
            cls: Type[InversionT],
            instrument: instruments.AbstractInstrument,
            scene_training: kgpy.solar.SpectralRadiance,
            scene_validation: kgpy.solar.SpectralRadiance,
            deprojections_training: kgpy.function.AbstractArray,
            deprojections_validation: kgpy.function.AbstractArray,
            instrument_inverse: None | keras.Sequential = None,
            num_divisions_fov: None | kgpy.vectors.Cartesian2D = None,
            epochs: int = 1000,
            # axs_spatial: Sequence[Sequence]
    ) -> InversionT:

        if num_divisions_fov is None:
            num_divisions_fov = kgpy.vectors.Cartesian2D(1, 1)

        average_output = 0
        # average_output = (np.median(scene_training.output) + np.median(scene_validation.output)) / 2

        width_output_training = np.percentile(scene_training.output, 99)
        # width_output_training = np.percentile(scene_training.output, 95) - np.percentile(scene_training.output, 5)
        width_output_validation = np.percentile(scene_validation.output, 99)
        # width_output_validation = np.percentile(scene_validation.output, 95) - np.percentile(scene_validation.output, 5)
        width_output = np.sqrt((np.square(width_output_training) + np.square(width_output_validation)) / 2)

        average_input = 0
        # average_input = (np.median(deprojections_training.output) + np.median(deprojections_validation.output)) / 2

        width_input_training = np.percentile(deprojections_training.output, 99)
        # width_input_training = np.percentile(deprojections_training.output, 95) - np.percentile(deprojections_training.output, 5)
        width_input_validation = np.percentile(deprojections_validation.output, 99)
        # width_input_validation = np.percentile(deprojections_validation.output, 95) - np.percentile(deprojections_validation.output, 5)
        width_input = np.sqrt((np.square(width_input_training) + np.square(width_input_validation)) / 2)

        print('average_input', average_input)
        print('average_output', average_output)
        print('width_input', width_input)
        print('width_output', width_output)

        scene_training.output = (scene_training.output - average_output) / width_output
        scene_validation.output = (scene_validation.output - average_output) / width_output
        deprojections_training.output = (deprojections_training.output - average_input) / width_input
        deprojections_validation.output = (deprojections_validation.output - average_input) / width_input

        scene_training.output = scene_training.output.add_axes(['channel'])
        scene_validation.output = scene_validation.output.add_axes(['channel'])

        axes_ordered = ['time', 'wavelength_offset', 'helioprojective_x', 'helioprojective_y', 'channel']

        shape_scene_training = {ax: scene_training.output.shape[ax] for ax in axes_ordered}
        shape_scene_validation = {ax: scene_validation.output.shape[ax] for ax in axes_ordered}
        shape_deprojections_training = {ax: deprojections_training.output.shape[ax] for ax in axes_ordered}
        shape_deprojections_validation = {ax: deprojections_validation.output.shape[ax] for ax in axes_ordered}

        scene_training.output = scene_training.output.broadcast_to(shape_scene_training)
        scene_validation.output = scene_validation.output.broadcast_to(shape_scene_validation)
        deprojections_training.output = deprojections_training.output.broadcast_to(shape_deprojections_training)
        deprojections_validation.output = deprojections_validation.output.broadcast_to(shape_deprojections_validation)

        print('scene_training', scene_training.output.shape)
        print('scene_validation', scene_validation.output.shape)
        print('deprojections_training', deprojections_training.output.shape)
        print('deprojections_validation', deprojections_validation.output.shape)

        num_channels = deprojections_training.shape['channel']

        if instrument_inverse is None:
            instrument_inverse = cls.instrument_inverse_initial(
                # input_shape=(num_channels, None, None, None),
                input_shape=(None, None, None, num_channels),
                n_filters=64,
                kernel_size=11,
                # kernel_size=scene_training.shape['wavelength_offset'],
                growth_factor=1,
                dropout_rate=0.1,
            )

        instrument_inverse.summary()

        instrument_inverse.optimizer = keras.optimizers.Nadam(
            learning_rate=5e-6,
        )

        # instrument_inverse.optimizer.lr = 2e-5
        # instrument_inverse.optimizer.beta_1 = 0.8
        print('optimizer', instrument_inverse.optimizer)
        print('learning_rate', instrument_inverse.optimizer.lr)
        print('beta_1', instrument_inverse.optimizer.beta_1)
        # print('momentum', instrument_inverse.optimizer.momentum)


        tensorboard_dir = pathlib.Path(__file__).parent / 'logs'
        callback_tensorboard = keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir / time.strftime("%Y%m%d-%H%M%S"),
            histogram_freq=0,
            write_graph=False,
            write_images=False,
        )

        checkpoint_filepath = pathlib.Path(__file__).parent / 'checkpoints'
        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        )

        callback_early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=1,
            restore_best_weights=True,
        )

        kwargs_fit = dict(
            batch_size=2,
            epochs=epochs,
            verbose=2,
            callbacks=[
                callback_tensorboard,
                # callback_checkpoint,
                callback_early_stopping,
            ],
            shuffle=True,
        )

        # for epoch in range(epochs):
        #     for index_batch in range(scene_training.output.shape['time']):
        #         instrument_inverse.train_on_batch(
        #             x=
        #         )
        #         instrument_inverse.loss()

        x_training = cls._output_divided(deprojections_training.output, num_divisions_fov).array
        x_validation = cls._output_divided(deprojections_validation.output, num_divisions_fov).array
        y_training = cls._output_divided(scene_training.output, num_divisions_fov).array
        y_validation = cls._output_divided(scene_validation.output, num_divisions_fov).array

        # history_identity = instrument_inverse.fit(
        #     x=np.broadcast_to(y_training, x_training.shape),
        #     y=y_training,
        #     validation_data=(
        #         np.broadcast_to(y_validation, x_validation.shape),
        #         y_validation,
        #     ),
        #     **kwargs_fit,
        # )
        # keras.backend.clear_session()
        #
        # print('sleeping')
        # time.sleep(5)
        # print('done sleeping')

        try:
            history_final = instrument_inverse.fit(
                x=x_training,
                y=y_training,
                validation_data=(
                    x_validation,
                    y_validation,
                ),
                **kwargs_fit,
            )
        except KeyboardInterrupt:
            history_final = instrument_inverse.history

        history = TrainingHistory(
            loss_training=np.concatenate([
                # history_identity.history['loss'],
                history_final.history['loss']
            ]),
            loss_validation=np.concatenate([
                # history_identity.history['val_loss'],
                history_final.history['val_loss'],
            ]),
        )


        return cls(
            instrument=instrument,
            instrument_inverse=instrument_inverse,
            history=history,
            average_input=average_input,
            average_output=average_output,
            width_input=width_input,
            width_output=width_output,
            num_divisions_fov=num_divisions_fov,
        )

    @staticmethod
    def instrument_inverse_initial(
            input_shape: tuple[int | None, int | None, int | None, int | None],
            n_filters: int = 32,
            kernel_size: int = 7,
            growth_factor: int = 2,
            alpha: float = 0.1,
            dropout_rate: float = 0.01,
    ) -> keras.Sequential:
        print('kernel_size', kernel_size)
        layers = [
            keras.layers.Conv3D(
                filters=n_filters * growth_factor ** 0,
                kernel_size=kernel_size,
                strides=growth_factor,
                padding='same',
                # data_format='channels_first',
                input_shape=input_shape,
                # kernel_regularizer=keras.regularizers.L2(1e-6),
                activation=keras.layers.LeakyReLU(alpha),
            ),
            keras.layers.Dropout(dropout_rate),
            # keras.layers.AvgPool3D(
            #     pool_size=(5, 1, 1),
            #     padding='same',
            # ),
            keras.layers.Conv3D(
                filters=n_filters * growth_factor ** 1,
                kernel_size=kernel_size,
                strides=growth_factor,
                dilation_rate=2,
                # kernel_size=(1, kernel_size, kernel_size),
                # kernel_size=(kernel_size // 5, kernel_size, kernel_size),
                padding='same',
                # data_format='channels_first',
                # kernel_regularizer="l2",
                activation=keras.layers.LeakyReLU(alpha),
            ),
            keras.layers.Dropout(dropout_rate),
            # # keras.layers.AvgPool3D(
            #     pool_size=(5, 1, 1),
            #     padding='same',
            # ),
            # keras.layers.UpSampling3D(size=growth_factor),
            keras.layers.Conv3DTranspose(
                filters=n_filters * growth_factor ** 0,
                kernel_size=kernel_size,
                strides=growth_factor,
                dilation_rate=2,
                # kernel_size=(1, kernel_size, kernel_size),
                # kernel_size=(kernel_size // 5 // 5, kernel_size, kernel_size),
                padding='same',
                # data_format='channels_first',
                # kernel_regularizer="l2",
                activation=keras.layers.LeakyReLU(alpha),
            ),
            # keras.layers.Dropout(dropout_rate),
            # keras.layers.UpSampling3D(
            #     size=(5, 1, 1),
            #     # padding='same',
            # ),
            # keras.layers.UpSampling3D(size=growth_factor),
            keras.layers.Conv3DTranspose(
                # filters=n_filters * growth_factor ** 3,
                filters=1,
                kernel_size=kernel_size,
                strides=growth_factor,
                # kernel_size=(kernel_size // 5, kernel_size, kernel_size),
                padding='same',
                kernel_initializer="zeros",
                # data_format='channels_first',
                # kernel_regularizer="l2",
                # kernel_regularizer=keras.regularizers.L2(1e-5),
            ),
            # keras.layers.LeakyReLU(
            #     alpha=alpha,
            # ),
            # keras.layers.Dropout(dropout_rate),
            # keras.layers.UpSampling3D(
            #     size=(5, 1, 1),
            #     # padding='same',
            # ),
            # keras.layers.Conv3D(
            #     filters=1,
            #     kernel_size=kernel_size,
            #     padding='same',
            #     kernel_initializer="zeros",
            #     # data_format='channels_first',
            #     # kernel_regularizer=keras.regularizers.L2(1e-6),
            # ),
            # keras.layers.LeakyReLU(
            #     alpha=alpha,
            # ),
        ]

        net = keras.Sequential(layers=layers)

        net.compile(
            # optimizer=keras.optimizers.experimental.AdamW(learning_rate=1e-6),
            optimizer=keras.optimizers.Nadam(learning_rate=5e-6),
            # optimizer=keras.optimizers.SGD(learning_rate=1e-3, momentum=1e-3),
            loss='mse',
        )

        return net

    def __call__(
            self: InversionT,
            image: kgpy.function.Array[kgpy.optics.vectors.DispersionOffsetSpectralPositionVector, kgpy.labeled.Array],
    ) -> kgpy.function.Array[kgpy.optics.vectors.SpectralFieldVector, kgpy.labeled.Array]:

        image = image.copy()
        image.output = (image.output - self.average_input) / self.width_input

        axes_ordered = ['time', 'wavelength_offset', 'helioprojective_x', 'helioprojective_y', 'channel']

        shape_image = {ax: image.output.shape[ax] for ax in axes_ordered}

        image.output = image.output.broadcast_to(shape_image)

        image.output.array = self.instrument_inverse.predict(
            x=image.output.array,
            batch_size=1,
        )

        image.output = image.output * self.width_output + self.average_output

        return image


# @dataclasses.dataclass
# class DiffusionInversion(
#     kgpy.mixin.Pickleable,
#     abstractions.AbstractInversion,
# ):
#
#     instrument_inverse: keras.Sequential
#     history: TrainingHistory
#     # average_input: kgpy.labeled.AbstractArray
#     # average_output: kgpy.labeled.AbstractArray
#     width_input: kgpy.labeled.AbstractArray
#     # width_output: kgpy.labeled.AbstractArray
#
#     def __call__(
#
#             self: InversionT,
#             image: kgpy.function.Array[kgpy.optics.vectors.DispersionOffsetSpectralPositionVector, kgpy.labeled.Array],
#     ) -> kgpy.function.Array[kgpy.optics.vectors.SpectralFieldVector, kgpy.labeled.Array]:
#
#         image = image.copy()
#         image.output = image.output / self.width_input
#         # image.output = (image.output - self.average_input) / self.width_input
#
#         axes_ordered = ['time', 'wavelength_offset', 'helioprojective_x', 'helioprojective_y', 'channel']
#
#         shape_image = {ax: image.output.shape[ax] for ax in axes_ordered}
#
#         image.output = image.output.broadcast_to(shape_image)
#
#         image.output.array = self.instrument_inverse.predict(
#             x=image.output.array,
#             batch_size=1,
#         )
#
#         image.output = image.output * self.width_input
#
#         return image
#
#     @classmethod
#     def train(
#             cls: Type[InversionT],
#             instrument: instruments.AbstractInstrument,
#             scene: kgpy.solar.SpectralRadiance,
#             instrument_inverse: None | keras.Sequential = None,
#             epochs: int = 1000,
#     ) -> InversionT:
#
#         axes_ordered = ['time', 'wavelength_offset', 'helioprojective_x', 'helioprojective_y', 'channel']
#
#         scene.output = scene.output.add_axes(["channel"])
#         scene.output = scene.output.broadcast_to({ax: scene.shape[ax] for ax in axes_ordered})
#
#         width_scene = np.percentile(scene.output, 99)
#         scene.output = scene.output / width_scene
#
#         num_frames_per_dataset = scene.shape['time'] // 2
#
#         slice_training = dict(time=slice(num_frames_per_dataset, None))
#         slice_validation = dict(time=slice(None, num_frames_per_dataset))
#
#         overlappogram = instrument(
#             scene=scene,
#             axis_wavelength='wavelength_offset',
#             axis_field=['helioprojective_x', 'helioprojective_y'],
#             wavelength_sum=False,
#         )
#         print("overlappogram", overlappogram.shape)
#         print("overlappogram.output", overlappogram.output.shape)
#
#         if instrument_inverse is None:
#             instrument_inverse = cls.instrument_inverse_initial(
#                 # input_shape=(num_channels, None, None, None),
#                 input_shape=(None, None, None, overlappogram.shape["channel"]),
#                 n_filters=8,
#                 # kernel_size=25,
#                 kernel_size=scene.shape['wavelength_offset'],
#             )
#
#         callback_early_stopping = keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=100,
#             verbose=1,
#             restore_best_weights=True,
#         )
#
#         kernel_size = 1
#
#         while True:
#
#             print("new diffusion step")
#             print("kernel_size", kernel_size)
#
#             kernel_shape = overlappogram.shape
#             for ax in kernel_shape:
#                 if ax == "wavelength_offset":
#                     kernel_shape[ax] = kernel_size
#                 else:
#                     kernel_shape[ax] = 1
#
#             print("kernel_shape", kernel_shape)
#
#             kernel = np.ones(tuple(kernel_shape.values()))
#
#             overlappogram_convolved = scipy.ndimage.convolve(
#                 input=overlappogram.output.array,
#                 weights=kernel,
#                 mode="constant",
#             )
#             norm = scipy.ndimage.convolve(
#                 input=np.ones_like(overlappogram.output.array),
#                 weights=kernel,
#                 mode="constant",
#             )
#             overlappogram_convolved = kgpy.function.Array(
#                 input=overlappogram.input,
#                 output=kgpy.labeled.Array(overlappogram_convolved / norm, axes=overlappogram.output.axes),
#             )
#
#             deprojection = instrument.deproject(
#                 image=overlappogram_convolved,
#             )
#             deprojection = deprojection.interp_linear(
#                 input_new=kgpy.optics.vectors.SpectralFieldVector(
#                     wavelength=scene.input.wavelength,
#                     field_x=scene.input.field_x,
#                     field_y=scene.input.field_y,
#                 ),
#                 axis=['wavelength_offset', 'detector_x', 'detector_y'],
#             )
#             deprojection.output = deprojection.output[dict(angle_input=0)]
#             deprojection.output = deprojection.output.broadcast_to({ax: deprojection.shape[ax] for ax in axes_ordered})
#
#             print("deprojection", deprojection.shape)
#             print("deprojection", deprojection.output.shape)
#
#             try:
#                 history = instrument_inverse.fit(
#                     x=deprojection.output[slice_training].array,
#                     y=scene.output[slice_training].add_axes(["channel"]).array,
#                     validation_data=(
#                         deprojection.output[slice_validation].array,
#                         scene.output[slice_validation].add_axes(["channel"]).array,
#                     ),
#                     batch_size=2,
#                     epochs=epochs,
#                     verbose=2,
#                     callbacks=[
#                         callback_early_stopping,
#                     ],
#                     shuffle=True,
#                 )
#             except KeyboardInterrupt:
#                 history = instrument_inverse.history
#                 break
#
#             if (kernel_size / 2) > scene.shape["wavelength_offset"]:
#                 print("kernel size is larger than wavelength axis, exiting")
#                 break
#
#             # kernel_size *= 2
#             kernel_size += 1
#
#         return cls(
#             instrument=instrument,
#             instrument_inverse=instrument_inverse,
#             history=TrainingHistory(
#                 loss_training=history.history["loss"],
#                 loss_validation=history.history["val_loss"]
#             ),
#             width_input=width_scene,
#         )
#
#     @staticmethod
#     def instrument_inverse_initial(
#             input_shape: tuple[int | None, int | None, int | None, int | None],
#             n_filters: int = 32,
#             kernel_size: int = 7,
#             growth_factor: int = 2,
#             alpha: float = 0.1,
#             dropout_rate: float = 0.01,
#             input_mean: float = 0,
#             input_variance: float = 1,
#     ) -> keras.Sequential:
#         print('kernel_size', kernel_size)
#         layers = [
#             keras.layers.Conv3D(
#                 filters=n_filters,
#                 kernel_size=kernel_size,
#                 padding='same',
#                 # data_format='channels_first',
#                 input_shape=input_shape,
#                 # kernel_regularizer="l2",
#             ),
#             keras.layers.LeakyReLU(
#                 alpha=alpha,
#             ),
#             # keras.layers.AvgPool3D(
#             #     pool_size=(5, 1, 1),
#             #     padding='same',
#             # ),
#             keras.layers.Conv3D(
#                 filters=n_filters,
#                 kernel_size=kernel_size,
#                 # kernel_size=(kernel_size // 5, kernel_size, kernel_size),
#                 padding='same',
#                 # data_format='channels_first',
#                 # kernel_regularizer="l2",
#             ),
#             keras.layers.LeakyReLU(
#                 alpha=alpha,
#             ),
#             # keras.layers.AvgPool3D(
#             #     pool_size=(5, 1, 1),
#             #     padding='same',
#             # ),
#             keras.layers.Conv3DTranspose(
#                 filters=n_filters,
#                 kernel_size=kernel_size,
#                 # kernel_size=(kernel_size // 5 // 5, kernel_size, kernel_size),
#                 padding='same',
#                 # data_format='channels_first',
#                 # kernel_regularizer="l2",
#             ),
#             keras.layers.LeakyReLU(
#                 alpha=alpha,
#             ),
#             # keras.layers.UpSampling3D(
#             #     size=(5, 1, 1),
#             #     # padding='same',
#             # ),
#             keras.layers.Conv3DTranspose(
#                 filters=n_filters,
#                 kernel_size=kernel_size,
#                 # kernel_size=(kernel_size // 5, kernel_size, kernel_size),
#                 padding='same',
#                 # data_format='channels_first',
#                 # kernel_regularizer="l2",
#             ),
#             keras.layers.LeakyReLU(
#                 alpha=alpha,
#             ),
#             # keras.layers.UpSampling3D(
#             #     size=(5, 1, 1),
#             #     # padding='same',
#             # ),
#             keras.layers.Conv3D(
#                 filters=1,
#                 kernel_size=kernel_size,
#                 padding='same',
#                 # data_format='channels_first',
#                 # kernel_regularizer="l2",
#             ),
#             # keras.layers.LeakyReLU(
#             #     alpha=alpha,
#             # ),
#         ]
#
#         net = keras.Sequential(layers=layers)
#
#         net.compile(
#             optimizer=keras.optimizers.Nadam(learning_rate=5e-7),
#             # optimizer=keras.optimizers.SGD(learning_rate=1e-3),
#             loss='mse',
#         )
#
#         net.summary()
#
#         return net
#

import astropy.units as u
import numpy as np
import scipy.ndimage
import typing as typ
import time
from skimage.transform import rotate
import dataclasses

__all__ = ['model', 'deproject']


def min_projection_shape(
        cube: np.ndarray,
        spectral_order: int,
        x_axis: int = ~1,
        w_axis: int = ~0,
) -> typ.Tuple[int, ...]:
    """
    Compute the minimum output shape.
    :param cube: `np.ndarray` of data
    :param spectral_order: integer, the spectral order for the data in `cube`
    :param x_axis:
    :param w_axis:
    :return:
    """
    psh = list(cube.shape)
    psh[x_axis] += psh[w_axis] * spectral_order
    psh[w_axis] = 1
    return tuple(psh)


def model(
        cube: np.ndarray,
        projection_azimuth: u.Quantity,
        spectral_order: int,
        projection_shape: typ.Tuple[int, ...] = None,
        cube_spatial_offset: typ.Tuple[int, int] = (0, 0),
        projection_spatial_offset: typ.Tuple[int, int] = (0, 0),
        x_axis: int = ~2,
        y_axis: int = ~1,
        w_axis: int = ~0,
        rotation_kwargs: typ.Dict = None
) -> np.ndarray:
    """
    Model is a basic forward model of CT imaging spectrograph.
    :param cube: 'slabs' which is a `np.ndarray` which have at least an x, y, and wavelength axis.
    :param projection_azimuth: scalar angle describing dispersion direction in the `data` array.
    :param spectral_order: number of indices the array shifted per wavelength bin.
    :param projection_shape: Desired shape of output array, as tuple of integers
    :param projection_spatial_offset:
    :param cube_spatial_offset:
    :param x_axis: axis of the data slabs representing x-axis
    :param y_axis: axis of the data slabs representing y-axis
    :param w_axis: axis of the data slabs representing wavelength axis
    :param rotation_kwargs: kwargs for `scipy.ndimage.rotate` to be used during rotation portion of forward model
    :return: list of arrays to which the forward model has been applied
    """

    if rotation_kwargs is None:
        rotation_kwargs = {'reshape': False, 'prefilter': False, 'order': 3, 'mode': 'nearest', }

    cube = cube.copy()
    az = projection_azimuth.to_value(u.deg)

    tsh = min_projection_shape(cube, spectral_order, x_axis, w_axis)
    if projection_shape is None:
        projection_shape = tsh

    p_data = np.zeros(projection_shape)

    px, py = projection_spatial_offset
    qx, qy = cube_spatial_offset

    ssh = list(cube.shape)
    x, y, l = np.meshgrid(np.arange(ssh[x_axis]), np.arange(ssh[y_axis]), np.arange(ssh[w_axis]))
    x, y, l = x.flatten(), y.flatten(), l.flatten()

    # rotated_cube = scipy.ndimage.rotate(
    #     input=cube.copy(),
    #     angle=az,
    #     axes=(x_axis, y_axis),
    #     **rotation_kwargs
    # )
    rotated_cube = rotate(
        image=cube.copy(),
        angle=az,
    )

    ssh[x_axis] += np.abs(spectral_order) * ssh[w_axis]
    shifted_cube = np.zeros(ssh)

    out_sl = [slice(None)] * shifted_cube.ndim
    in_sl = [slice(None)] * rotated_cube.ndim
    out_sl[x_axis], out_sl[y_axis], out_sl[w_axis] = x + spectral_order * l, y, l
    in_sl[x_axis], in_sl[y_axis], in_sl[w_axis] = x, y, l

    # Making a little bit future proof:
    out_sl = tuple(out_sl)
    in_sl = tuple(in_sl)

    shifted_cube[out_sl] = rotated_cube[in_sl]

    out_sl = [slice(None)] * p_data.ndim
    in_sl = [slice(None)] * shifted_cube.ndim
    tx = px - qx
    ty = py - qy

    out_sl[x_axis] = slice(max(0, tx), min(p_data.shape[x_axis], tx + ssh[x_axis]))
    out_sl[y_axis] = slice(max(0, ty), min(p_data.shape[y_axis], ty + ssh[y_axis]))
    in_sl[x_axis] = slice(max(0, -tx), min(ssh[x_axis], -tx + p_data.shape[x_axis]))
    in_sl[y_axis] = slice(max(0, -ty), min(ssh[y_axis], -ty + p_data.shape[y_axis]))

    # Making a little bit future proof:
    # Using the tuple loophole
    out_sl = tuple(out_sl)
    in_sl = tuple(in_sl)

    p_data[out_sl] = np.sum(shifted_cube, axis=w_axis, keepdims=True)[in_sl]
    return p_data


def deproject(
        projection: np.ndarray,
        projection_azimuth: u.Quantity,
        spectral_order: int,
        cube_shape: typ.Tuple[int, ...] = None,
        cube_spatial_offset: typ.Tuple[int, int] = (0, 0),
        projection_spatial_offset: typ.Tuple[int, int] = (0, 0),
        x_axis: int = ~2,
        y_axis: int = ~1,
        w_axis: int = ~0,
        rotation_kwargs: typ.Dict = None
) -> np.ndarray:
    if rotation_kwargs is None:
        rotation_kwargs = {'reshape': False, 'prefilter': False, 'order': 3, 'mode': 'nearest', }

    csh = list(projection.shape)
    # csh = list(cube_shape)
    csh[w_axis] = cube_shape[w_axis]

    projection = np.broadcast_to(projection, csh, subok=True)

    shifted_projection = np.zeros(cube_shape)

    px, py = cube_spatial_offset
    qx, qy = projection_spatial_offset

    out_sl = [slice(None)] * shifted_projection.ndim
    in_sl = [slice(None)] * projection.ndim
    tx = px - qx
    ty = py - qy

    out_sl[x_axis] = slice(max(0, tx), min(shifted_projection.shape[x_axis], tx + csh[x_axis]))
    out_sl[y_axis] = slice(max(0, ty), min(shifted_projection.shape[y_axis], ty + csh[y_axis]))
    in_sl[x_axis] = slice(max(0, -tx), min(csh[x_axis], -tx + shifted_projection.shape[x_axis]))
    in_sl[y_axis] = slice(max(0, -ty), min(csh[y_axis], -ty + shifted_projection.shape[y_axis]))

    shifted_projection[tuple(out_sl)] = projection[in_sl]
    backprojected_cube = np.zeros_like(shifted_projection)

    ssh = list(cube_shape)
    x, y, l = np.meshgrid(np.arange(ssh[x_axis]), np.arange(ssh[y_axis]), np.arange(ssh[w_axis]), copy=False,
                          sparse=False)

    x, y, l = x.flatten(), y.flatten(), l.flatten()

    out_sl = [slice(None)] * shifted_projection.ndim
    in_sl = [slice(None)] * shifted_projection.ndim
    out_sl[x_axis], out_sl[y_axis], out_sl[w_axis] = x - spectral_order * (l), y, l
    in_sl[x_axis], in_sl[y_axis], in_sl[w_axis] = x, y, l

    backprojected_cube[tuple(out_sl)] = shifted_projection[in_sl]
    del shifted_projection
    del x, y, l

    az = -1 * projection_azimuth.to_value(u.deg)

    # backprojected_cube = scipy.ndimage.rotate(
    #     input=backprojected_cube,
    #     angle=az,
    #     axes=(x_axis, y_axis),
    #     **rotation_kwargs
    # )


    backprojected_cube = rotate(
        image=backprojected_cube,
        angle=az,
    )

    return backprojected_cube

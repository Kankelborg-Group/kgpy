import numpy as np
import typing as typ

__all__ = ['rebin']


def rebin(arr: np.ndarray, scale_dims: typ.Tuple[int,...]):
    """
    Increases the size of an array by scale_dims in each i dimension by repeating each value scale_dims[i] times along
     that axis
    """
    new_arr = np.broadcast_to(arr,scale_dims+arr.shape)
    start_axes = np.arange(arr.ndim)
    new_axes = 2*start_axes+1
    new_arr = np.moveaxis(new_arr,start_axes,new_axes)

    new_shape = np.array(arr.shape)*np.array(scale_dims)
    new_arr = np.reshape(new_arr,new_shape)
    return new_arr

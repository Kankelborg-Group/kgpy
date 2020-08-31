import numpy as np

def rebin(arr: np.ndarray,scale_dims: tuple):

    """
    Increases the size of an array by scale_dims in each i dimension by repeating each value scale_dims[i] times along
     that axis
    """
    new_arr = np.broadcast_to(arr,scale_dims+arr.shape)
    start_axes = np.arange(arr.ndim)
    new_axes = 2*start_axes+1
    new_arr = np.moveaxis(new_arr,tuple(start_axes),tuple(new_axes))

    new_shape = [arr.shape[i]*scale_dims[i] for i in range(arr.ndim)]
    new_arr = np.reshape(new_arr,tuple(new_shape))
    return new_arr

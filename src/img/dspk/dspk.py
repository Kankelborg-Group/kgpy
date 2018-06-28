
from src.img.dspk import dspk_ndarr

def dspk_3D(data, lower_thresh=0.01, upper_thresh=0.99, kernel_shape=(25,25,25)):

    lmed = dspk_ndarr(data, lower_thresh, upper_thresh, kernel_shape[0], kernel_shape[1], kernel_shape[2],-200)

    return lmed
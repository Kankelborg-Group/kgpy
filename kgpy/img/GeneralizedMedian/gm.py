import numpy as np
from scipy import signal
from scipy import linalg
def gm(image):
    """
    Compute the generalized median (GM) of a 2D numpy array (image). The GM
    is a robust measure of central tendency. The GM is less susceptible to 
    outliers, noise, background and negative pixel values than the centroid.
    Like the centroid, however, the GM can be computed to sub-pixel accuracy.
    This implementation takes advantage of FFT convolution for efficiency.
    
    The result is a numpy array given in pixel coordinates, with [0.,0.] corresponding
    to the center of image[0,0]
    
    2018-Oct-27  C. Kankelborg 
    """
    (Nx,Ny) = image.shape()
    x,y = np.meshgrid(np.arange(-Nx:Nx+1),np.arange(-Ny:Ny+1))
    r = np.sqrt(x**2 + y**2) # Convolution kernel
    wmap = signal.fftconvolve(image, r, mode='same')
    
    (im,jm) = np.unravel_index(np.argmin(wmap), wmap.shape)

    # Fit 3x3 neighborhood around the minimum to get subpixel accuracy
    neighborhood = wmap[im-1:im+2, jm-1:jm+2]
    X,Y = np.meshgrid([-1,0,1],[-1,0,1])
    Xc = X.flatten() # X squished into a column vector
    Yc = Y.flatten() # Y squished into a column vector
    Cc = np.ones(Xc.shape[0]) # Constant column vector
    A = np.c_[Cc, Xc, Yc, Xc*Yc, Xc**2, Yc**2]
    coeff,_,_,_ = linalg.lstsq(A, neighborhood)
    # From this, calculate offsets from im,jm
    delta_x = 
    delta_y = 
    
    # Calculate and return rm
    return (im+delta_x, jm+delta_y)
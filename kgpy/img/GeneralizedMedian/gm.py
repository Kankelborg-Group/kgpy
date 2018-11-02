import numpy as np
from scipy import signal
from scipy import linalg
def gm2d(image):
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
    Ny,Nx = image.shape
    x,y = np.meshgrid(np.arange(-Nx,Nx+1),np.arange(-Ny,Ny+1))
    r = np.sqrt(x**2 + y**2) # Convolution kernel
    wmap = signal.fftconvolve(image, r, mode='same')    
    (im,jm) = np.unravel_index(np.argmin(wmap), wmap.shape)

    # Fit 3x3 neighborhood around the minimum to get subpixel accuracy
    neighborhood = wmap[im-1:im+2, jm-1:jm+2]
    X,Y = np.meshgrid([-1,0,1],[-1,0,1]) # Neighborhood coordinates
    Xc = X.flatten() # X squished into a column vector
    Yc = Y.flatten() # Y squished into a column vector
    Cc = np.ones(Xc.shape[0]) # Constant column vector
    A = np.c_[Cc, Xc, Yc, Xc*Yc, Xc**2, Yc**2] # Matrix to be solved
    C = linalg.lstsq(A, neighborhood.flatten())[0] # Array of fit coefficients
    # From this, calculate offsets from im,jm.
    # I am simply using the analytic solution for the minimum of
    # C[0] + C[1]*X + C[2]*Y + C[3]*X*Y + C[4]*X**2 + C[5]*Y**2.
    denominator = C[3]**2 - 4*C[4]*C[5]
    delta_x = -(C[2]*C[3] - 2*C[5]*C[1]) / denominator
    delta_y = -(C[1]*C[3] - 2*C[4]*C[2]) / denominator
    
    meanabsrad = wmap[im,jm]/np.sum(image)
    
    # Calculate and return rm
    return (jm,im,delta_x,delta_y,meanabsrad)
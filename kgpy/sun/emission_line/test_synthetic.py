import numpy as np
from math import exp as exp
def synthetic(A,x0,w,N):
    '''
    Gaussian of the form f(x)=exp( -(x-x0)^2 / w^2 )
    :param A: Height of the Gaussian
    :param x0: Middle of the Gaussian
    :param w: the standard variation of the Gaussian
    :param N: The number of entries.
    :return: Produces a NumPy array that holds a Gaussian
    '''
    xlow = x0 - 5*w
    step=5*w/(N)
    gaussian=np.zeros((N,2))
    for j in range(1,N):
        xj=xlow+(step*j)
        fj= A*exp( -((x0-xj)**2 )/ (w**2))
        gaussian[j] = ([fj, xj])
        return gaussian
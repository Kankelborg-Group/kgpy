import numpy as np


def synthetic(A: float, x0: float, w: float, N: int):
    '''
    Gaussian of the form f(x)=exp( -(x-x0)^2 / w^2 )
    :param A: Height of the Gaussian
    :param x0: Middle of the Gaussian
    :param w: the standard variation of the Gaussian
    :param N: The number of points in the Gaussian
    :return: Produces a NumPy array that holds a Gaussian
    '''
    # xlow = x0 - 5*w
    # step=5*w/(N)
    # gaussian=np.zeros((N,2))
    # for j in range(1,N):
    #     xj=xlow+(step*j)
    #     fj= a*exp( -((x0-xj)**2 )/ (w**2))
    #     gaussian[j] = ([fj, xj])
    #     return gaussian
    #
    # Arguments to function
    x_min = -10
    x_max = 10

    x = np.linspace(x_min, x_max, N)

    X = x - x0

    y = A * np.exp(- X * X / (w * w))

    return x, y

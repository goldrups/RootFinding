"""
Author: Sam Goldrup
naive method for approximating a univariate continuous 
function on arbitrary interival using the chebyshev
polynomial basis
"""

import numpy as np
from numpy.fft import fft,fft2
from matplotlib import pyplot as plt
from yroots.utils import transform


def interval_approx_1d(f,a,b,deg):
    """
    returns the coefficients of the chebyshev basis

    f: R-->R function to approximate

    a: the lower bound of the interval
    b: the upper bound of the interval
    deg: degree of approximation
    """
    cheby_ext = transform(np.cos((np.pi*np.arange(2*deg))/deg),a,b)
    values = f(cheby_ext) #interpolating y values
    coeffs = np.real(fft(values/deg)) #fourier coeffs
    coeffs[0] = coeffs[0]/2
    coeffs[deg] = coeffs[deg]/2

    return coeffs[:deg+1] #deg=n, then we get the n+1 first coeffs

def interval_approx_2d(f,a,b,deg):
    """
    Gets ...

    f:R^2 --> R
    a: array
    lower bound on the interval
    b: array
    upper bound on the interval
    deg: array
    degree of approximation in each dimension
    """
    dim = len(a)
    if dim != len(b):
        raise ValueError("Interval dimensions must be the same!")
    
    cheby_ext_1 = transform(np.cos((np.pi*np.arange(2*deg[0]))/deg[0]),a[0],b[0])
    cheby_ext_2 = transform(np.cos((np.pi*np.arange(2*deg[1]))/deg[1]),a[1],b[1])

    grid_pts = np.meshgrid(cheby_ext_1,cheby_ext_2)
    values_grid = f(grid_pts)
    coeffs = np.real(fft2(values_grid)/(deg[0]*deg[1]))
    coeffs[0,0] = coeffs[0,0]/2
    coeffs[0,deg[0]] = coeffs[0,deg[0]]/2 #one's and zeros as indices maybe fixed this
    coeffs[deg[1],0] = coeffs[deg[1],0]/2
    coeffs[deg[1],deg[0]] = coeffs[deg[1],deg[0]]/2

    coeffs = np.array(coeffs)
    print(np.shape(coeffs))
    return coeffs[:deg[1]+1,:deg[0]+1]
    #checkout interval_approx_slicers



    


if __name__ == "__main__":
    g = lambda x: x[0]**2+2*x[1]+3*x[1]**3+4*x[0]**11
    a,b=[-0.1,0.2],[0.2,0.6]
    deg = [4,4]
    coeffs = interval_approx_2d(g,a,b,deg)
    print(np.shape(coeffs))
    # f = np.polynomial.chebyshev.Chebyshev(linear_combo,[a,b])
    # domain = np.linspace(a,b,1001)
    # plt.plot(domain,g(domain),label='g',color='r')
    # plt.plot(domain,f(domain),label='f',color='g')
    # plt.legend()
    # plt.show()







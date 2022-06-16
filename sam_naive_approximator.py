"""
Author: Sam Goldrup
naive method for approximating a univariate continuous 
function on arbitrary interival using the chebyshev
polynomial basis
"""

import numpy as np
from numpy.fft import fft,fft2,fftn
from matplotlib import pyplot as plt
from yroots.utils import transform
from yroots.subdivision import interval_approx_slicers, chebyshev_block_copy


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

def interval_approx_nd(f,a,b,deg):
    """
    Gets ...

    f:R^2 --> R
    a: array
    lower bound on the interval
    b: array
    upper bound on the interval
    deg: int
    degree of approximation for all the dimensions
    """
    if len(a) != len(b):
        raise ValueError("dimension mismatch oh nooo")
    dim = len(a)

    cheb_vals = np.cos(np.arange(deg+1)*np.pi/deg)
    cheb_grid = np.meshgrid(*([cheb_vals]*dim),indexing='ij')
    flatten = lambda x: x.flatten()
    cheby_pts = np.column_stack(tuple(map(flatten, cheb_grid)))

    cheb_pts = transform(cheby_pts,a,b)
    #print(len(cheb_pts.T))
    values_block = f(*cheb_pts.T).reshape(*([deg+1]*dim))

    #values = chebyshev_block_copy(values_block)
    values = values_block

    # cheby_ext_1 = transform(np.cos((np.pi*np.arange(2*deg))/deg),a[i],b[i])
    # cheby_ext_2 = transform(np.cos((np.pi*np.arange(2*deg))/deg),a[1],b[1])

    # grid_pts = 
    # values_grid = f(grid_pts)
    coeffs = np.real(fft2(values)/(deg**dim))
    x0_slicer, deg_slicer, slices, rescale = interval_approx_slicers(dim,deg)
    coeffs = fftn(values/rescale).real

    for x0sl, degsl in zip(x0_slicer, deg_slicer):
        # halve the coefficients in each slice
        coeffs[x0sl] /= 2
        coeffs[degsl] /= 2

    return coeffs[tuple(slices)]
    #checkout interval_approx_slicers


if __name__ == "__main__":
    g = lambda x,y: ((90000*y**10 + (-1440000)*y**9 + (360000*x**4 + 720000*x**3 + 504400*x**2 + 144400*x + 9971200)*(y**8) +
                ((-4680000)*x**4 + (-9360000)*x**3 + (-6412800)*x**2 + (-1732800)*x + (-39554400))*(y**7) + (540000*x**8 +
                2160000*x**7 + 3817600*x**6 + 3892800*x**5 + 27577600*x**4 + 51187200*x**3 + 34257600*x**2 + 8952800*x + 100084400)*(y**6) +
                ((-5400000)*x**8 + (-21600000)*x**7 + (-37598400)*x**6 + (-37195200)*x**5 + (-95198400)*x**4 +
                (-153604800)*x**3 + (-100484000)*x**2 + (-26280800)*x + (-169378400))*(y**5) + (360000*x**12 + 2160000*x**11 +
                6266400*x**10 + 11532000*x**9 + 34831200*x**8 + 93892800*x**7 + 148644800*x**6 + 141984000*x**5 + 206976800*x**4 +
                275671200*x**3 + 176534800*x**2 + 48374000*x + 194042000)*(y**4) + ((-2520000)*x**12 + (-15120000)*x**11 + (-42998400)*x**10 +
                (-76392000)*x**9 + (-128887200)*x**8 + (-223516800)*x**7 + (-300675200)*x**6 + (-274243200)*x**5 + (-284547200)*x**4 +
                (-303168000)*x**3 + (-190283200)*x**2 + (-57471200)*x + (-147677600))*(y**3) + (90000*x**16 + 720000*x**15 + 3097600*x**14 +
                9083200*x**13 + 23934400*x**12 + 58284800*x**11 + 117148800*x**10 + 182149600*x**9 + 241101600*x**8 + 295968000*x**7 +
                320782400*x**6 + 276224000*x**5 + 236601600*x**4 + 200510400*x**3 + 123359200*x**2 + 43175600*x + 70248800)*(y**2) +
                ((-360000)*x**16 + (-2880000)*x**15 + (-11812800)*x**14 + (-32289600)*x**13 + (-66043200)*x**12 + (-107534400)*x**11 +
                (-148807200)*x**10 + (-184672800)*x**9 + (-205771200)*x**8 + (-196425600)*x**7 + (-166587200)*x**6 + (-135043200)*x**5 +
                (-107568800)*x**4 + (-73394400)*x**3 + (-44061600)*x**2 + (-18772000)*x + (-17896000))*y + (144400*x**18 + 1299600*x**17 +
                5269600*x**16 + 12699200*x**15 + 21632000*x**14 + 32289600*x**13 + 48149600*x**12 + 63997600*x**11 + 67834400*x**10 +
                61884000*x**9 + 55708800*x**8 + 45478400*x**7 + 32775200*x**6 + 26766400*x**5 + 21309200*x**4 + 11185200*x**3 + 6242400*x**2 +
                3465600*x + 1708800)))
    a,b=np.array([-0.1,0.2]),np.array([0.2,0.6])
    deg = 3
    p = interval_approx_nd(g,a,b,deg)
    print(p)

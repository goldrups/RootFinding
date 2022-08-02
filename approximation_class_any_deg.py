from ast import AsyncFunctionDef
from yroots import eriks_code
from yroots.subdivision import solve
import numpy as np
from yroots.utils import transform, slice_top
from scipy.fftpack import fftn
from itertools import product
from time import time
from tests.chebfun2_suite import norm_pass_or_fail,residuals,residuals_pass_or_fail,verbose_pass_or_fail
from matplotlib import pyplot as plt

class M_maker:
    def __init__(self,f,a,b,guess_deg,return_inf_norm=False,rel_approx_tol=1.e-15, abs_approx_tol=1.e-12,force_deg=False):
        #print(deg)
        dim = len(a)
        if dim != len(b):
            raise ValueError("dimension mismatch")
        self.dim = dim
        self.f = f
        self.a = a
        self.b = b
        self.rel_approx_tol = rel_approx_tol
        self.abs_approx_tol = abs_approx_tol
        self.return_inf_norm = return_inf_norm
        self.values_block = None

        if force_deg:
            self.deg = guess_deg
        else:
            #print("this executed OMG")
            self.deg = self.find_good_deg(f,guess_deg,dim,a,b)
        #print(self.deg)

        if self.return_inf_norm == True:
            #print("will call it 0")
            self.M, self.inf_norm = self.interval_approximate_nd(self.f,self.a,self.b,self.deg,self.return_inf_norm, foreal=True)
            #print("called it 00")
            self.M2 = self.interval_approximate_nd(self.f,self.a,self.b,2*self.deg,self.return_inf_norm)[0]
            self.M2[slice_top(self.M.shape)] -= self.M
            self.err = np.sum(np.abs(self.M2))
            #print(self.err)
        else:
            #print("will call it 0")
            self.M = self.interval_approximate_nd(self.f,self.a,self.b,self.deg, foreal=True)
            #print("called it 00")
            self.M2 = self.interval_approximate_nd(self.f,self.a,self.b,2*self.deg)
            self.M2[slice_top(self.M.shape)] -= self.M
            self.err = np.sum(np.abs(self.M2))
            #print(self.err)


    def error_test(self,error,abs_approx_tol,rel_approx_tol,inf_norm): 
        """
        Determines whether the approximation is within the error tolerance

        Parameters
        ----------
        error: float
        The absolute value of the difference of the sum of abs values of M and M2
        rel_approx_tol: float
        some input I might want to cover my bases about
        abs_approx_tol: float
        some input I might want to cover my bases about
        inf_norm: float
        the sup norm on the approximation

        Returns
        -------
        Bool: if the error test has been passed or not
        """
        #print("welcome to approx class 2")
        return error < abs_approx_tol+rel_approx_tol*inf_norm

    def find_good_deg(self,f,deg,dim,a,b):
        """
        Finds the right degree with which to approximate on the interval.

        Parameters
        ----------
        f : function from R^n -> R
        The function to interpolate.
        deg : numpy array
        The degree of the interpolation in each dimension.
        dim: int
        Dimension
        a : numpy array
        The lower bound on the interval.
        b : numpy array
        The upper bound on the interval.

        Returns
        -------
        deg: the correct approximation degree
        """
        #print("finding a good degree")
        max_deg = {1: 100000, 2:1000, 3:9, 4:9, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2}

        coeff = self.interval_approximate_nd(f, a, b, deg)
        #print("called it 1")
        coeff2, inf_norm = self.interval_approximate_nd(f, a, b, deg*2, return_inf_norm=True)
        #print("called it 2")
        coeff2[slice_top(coeff.shape)] -= coeff
        self.err = np.sum(np.abs(coeff2))

        if deg >= max_deg[dim]: #might be able to stream line this if branch and the while loop
            print("already too much/enough deg")
            deg = max_deg[dim]
            return deg

        while deg < max_deg[dim]:
            print(deg)
            #print(self.err)
            if self.error_test(self.err,self.abs_approx_tol,self.rel_approx_tol,inf_norm):
                print("passed the test")
                break
            elif 2*deg > max_deg[dim]:
                print("maxxed out")
                deg = max_deg[dim]
                break
            else:
                print("failure and double")
                deg = int(2*deg)
                double_deg = int(2*deg)
                #print("will call it 3")
                coeff = self.interval_approximate_nd(f, a, b, deg)
                #print("will call it 4")
                coeff2, inf_norm = self.interval_approximate_nd(f, a, b, double_deg, return_inf_norm=True)
                coeff2[slice_top(coeff.shape)] -= coeff
                self.err = np.sum(np.abs(coeff2))
        
        return deg

    def interval_approximate_nd(self,f, a, b, deg, return_inf_norm=False, foreal=False):
        """Finds the chebyshev approximation of an n-dimensional function on an
        interval.

        Parameters
        ----------
        f : function from R^n -> R
            The function to interpolate.
        a : numpy array
            The lower bound on the interval.
        b : numpy array
            The upper bound on the interval.
        deg : numpy array
            The degree of the interpolation in each dimension. #Question THIS IS A NUMPY ARRAY
        return_inf_norm : bool
            whether to return the inf norm of the function

        Returns
        -------
        coeffs : numpy array
            The coefficient of the chebyshev interpolating polynomial.
        inf_norm : float
            The inf_norm of the function
        """
        #print("doing an approximation")
        dim = len(self.a)
        if dim != len(self.b):
            raise ValueError("Interval dimensions must be the same!")

        if hasattr(self.f,"evaluate_grid"):
            cheb_values = np.cos(np.arange(deg+1)*np.pi/deg) #simply executes the lines within the function instead of the function call
            chepy_pts =  np.column_stack([cheb_values]*dim)
            cheb_pts = transform(chepy_pts,a,b)
            self.values_block = f.evaluate_grid(cheb_pts)
        else:
            #print("deg is", deg)
            #print("and the type is...", type(deg))
            cheb_vals = np.cos(np.arange(deg+1)*np.pi/deg)
            cheb_grid = np.meshgrid(*([cheb_vals]*dim),indexing='ij')
            flatten = lambda x: x.flatten()
            cheby_pts = np.column_stack(tuple(map(flatten, cheb_grid)))
            cheb_pts = transform(cheby_pts,a,b)
            #print(cheb_pts)
            values_block = f(*cheb_pts.T).reshape(*([deg+1]*dim))
            #print(self.values_block)
        if foreal == True:
            self.values_block = values_block
        self.values = self.chebyshev_block_copy(values_block)

        if return_inf_norm:
            inf_norm = np.max(np.abs(self.values))

        x0_slicer, deg_slicer, slices, rescale = self.interval_approx_slicers(dim,deg)
        coeffs = fftn(self.values/rescale).real

        for x0sl, degsl in zip(x0_slicer, deg_slicer):
            # halve the coefficients in each slice
            coeffs[x0sl] /= 2
            coeffs[degsl] /= 2

        if return_inf_norm:
            return coeffs[tuple(slices)], inf_norm
        else:
            return coeffs[tuple(slices)]
    
    def chebyshev_block_copy(self,values_block):
        """This functions helps avoid double evaluation of functions at
        interpolation points. It takes in a tensor of function evaluation values
        and copies these values to a new tensor appropriately to prepare for
        chebyshev interpolation.

        Parameters
        ----------
        values_block : numpy array
        block of values from function evaluation
        Returns
        -------
        values_cheb : numpy array
        chebyshev interpolation values
        """
        #print("doing a block copy")
        #np.empty(tuple([2*deg])*dim, dtype=np.float64)
        dim = values_block.ndim
        deg = values_block.shape[0] - 1
        #values_cheb = values_arr(dim)
        values_cheb = np.empty(tuple([2*deg])*dim, dtype=np.float64) #self.values_cheb?
        block_slicers, cheb_slicers, slicer = self.block_copy_slicers(dim, deg)

        for cheb_idx, block_idx in zip(cheb_slicers, block_slicers):
            try:
                values_cheb[cheb_idx] = values_block[block_idx]
            except ValueError as e:
                if str(e)[:42] == 'could not broadcast input array from shape': 
                    #self.values_arr.memo[(dim, )] = np.empty(tuple([2*deg])*dim, dtype=np.float64) #I KNOW WHAT THIS DOES!
                    values_cheb = np.empty(tuple([2*deg])*dim, dtype=np.float64)
                    values_cheb[cheb_idx] = values_block[block_idx]
                else:
                    raise ValueError(e)
        return values_cheb[slicer]
    
    def block_copy_slicers(self,dim, deg):
        """Helper function for chebyshev_block_copy.
        Builds slice objects to index into the evaluation array to copy
        in preparation for the fft.

        Parameters
        ----------
        dim : int
            Dimension
        dim : int
            Degree of approximation

        Returns
        -------
        block_slicers : list of tuples of slice objects
            Slice objects used to index into the evaluations
        cheb_slicers : list of tuples of slice objects
            Slice objects used to index into the array we're copying evaluations to
        slicer : tuple of slice objets
            Used to index into the portion of that array we're using for the fft input
        """
        #print("getting slicers")
        block_slicers = []
        cheb_slicers = []
        full_arr_deg = 2*deg
        for block in product([False, True], repeat=dim):
            cheb_idx = [slice(0, deg+1)]*dim
            block_idx = [slice(0, full_arr_deg)]*dim
            for i, flip_dim in enumerate(block):
                if flip_dim:
                    cheb_idx[i] = slice(deg+1, full_arr_deg)
                    block_idx[i] = slice(deg-1, 0, -1)
            block_slicers.append(tuple(block_idx))
            cheb_slicers.append(tuple(cheb_idx))
        return block_slicers, cheb_slicers, tuple([slice(0, 2*deg)]*dim)

    def interval_approx_slicers(self,dim, deg):
        """Helper function for interval_approximate_nd. Builds slice objects to index
        into the output of the fft and divide some of the values by 2 and turn them into
        coefficients of the approximation.

        Parameters
        ----------
        dim : int
            The interpolation dimension.
        deg : int
            The interpolation degree. #SEE WE TAKE THIS AS A SCALAR

        Returns
        -------
        x0_slicer : list of tuples of slice objects
            Slice objects used to index into the the degree 1 monomials
        deg_slicer : list of tuples of slice objects
            Slice objects used to index into the the degree d monomials
        slices : tuple of slice objets
            Used to index into the portion of the array that are coefficients
        rescale : int
            amount to rescale the evaluations by in order to feed them into the fft
        """
        #print("getting helped")
        x0_slicer = [tuple([slice(None) if i != d else 0 for i in range(dim)])
                    for d in range(dim)]
        deg_slicer = [tuple([slice(None) if i != d else deg for i in range(dim)])
                    for d in range(dim)]
        slices = tuple([slice(0, deg+1)]*dim)
        return x0_slicer, deg_slicer, slices, deg**dim

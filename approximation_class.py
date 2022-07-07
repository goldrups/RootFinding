from yroots.subdivision import interval_approximate_nd
from yroots import eriks_code
import numpy as np
from yroots.utils import transform
from scipy.fftpack import fftn
from itertools import product

class M_maker:
    def __init__(self,f,a,b,deg):
        self.f = f
        self.a = a
        self.b = b
        self.deg = deg
        # self.values = []
        # self.values_block = []
        self.memoizer = {}

    def interval_approximate_nd(self,f, a, b, deg, return_inf_norm=False):
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
    
        dim = len(self.a)
        if dim != len(self.b):
            raise ValueError("Interval dimensions must be the same!")

        if hasattr(self.f,"evaluate_grid"):
            cheb_values = np.cos(np.arange(deg+1)*np.pi/deg) #simply executes the lines within the function instead of the function call
            chepy_pts =  np.column_stack([cheb_values]*dim)
            cheb_pts = transform(chepy_pts,a,b)
            self.values_block = f.evaluate_grid(cheb_pts)
        else:
            cheb_vals = np.cos(np.arange(deg+1)*np.pi/deg)
            cheb_grid = np.meshgrid(*([cheb_vals]*dim),indexing='ij')
            flatten = lambda x: x.flatten()
            cheby_pts = np.column_stack(tuple(map(flatten, cheb_grid)))
            cheb_pts = transform(cheby_pts,a,b)
            self.values_block = f(*cheb_pts.T).reshape(*([deg+1]*dim))

        self.values = self.chebyshev_block_copy(self.values_block)

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
        dim = values_block.ndim
        deg = values_block.shape[0] - 1
        values_cheb = self.values_arr(dim) #
        block_slicers, cheb_slicers, slicer = self.block_copy_slicers(dim, deg)

        for cheb_idx, block_idx in zip(cheb_slicers, block_slicers):
            try:
                values_cheb[cheb_idx] = values_block[block_idx]
            except ValueError as e:
                if str(e)[:42] == 'could not broadcast input array from shape':
                    self.values_arr.memo[(dim, )] = np.empty(tuple([2*deg])*dim, dtype=np.float64) #MEMO
                    values_cheb = self.values_arr(dim)
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

    def initialize_values_arr(self,dim, deg):
        """Helper function for chebyshev_block_copy.
        Initializes an array to use throughout the whole solve function.
        Builds one array corresponding to dim and deg that can be used for any
        block copy of degree less than deg

        Parameters
        ----------
        dim : int
            Dimension
        deg : int
            Degree

        Returns
        -------
        An empty numpy array that can be used to hold values for a chebyshev_block_copy
        of dimension dim degree < deg.
        """
        return np.empty(tuple([2*deg])*dim, dtype=np.float64)

    def values_arr(self,dim):
        """Helper function for chebyshev_block_copy.
        Finds the array initialized by initialize_values_arr for dimension dim.
        Assumes the degree of the approximation is less than the degree used for
        initialize_values_arr.

        Parameters
        ----------
        dim : int
            Dimension

        Returns
        -------
        An empty numpy array that can be used to hold values for a chebyshev_block_copy
        of dimension dim and degree less than the degree used for initialize_values_arr.
        """
        keys = tuple(self.initialize_values_arr.memo.keys()) #MEMO
        for idx, k in enumerate(keys):
            if k[0]==dim:
                break
        return self.initialize_values_arr.memo[keys[idx]] #MEMO

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
        x0_slicer = [tuple([slice(None) if i != d else 0 for i in range(dim)])
                    for d in range(dim)]
        deg_slicer = [tuple([slice(None) if i != d else deg for i in range(dim)])
                    for d in range(dim)]
        slices = tuple([slice(0, deg+1)]*dim)
        return x0_slicer, deg_slicer, slices, deg**dim



f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
g = lambda x,y: y-x**6
a,b = np.array([-1,-1]),np.array([1,1])
f_deg,g_deg = 4,6
Mf = interval_approximate_nd(f,a,b,f_deg)
Mg = interval_approximate_nd(g,a,b,g_deg)

err_f = Mf[0,0] + Mf[0,1] + Mf[1,0]
err_g = Mg[0,0] + Mg[0,1] + Mg[1,0]


roots = eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g]))
print(roots)
from yroots import eriks_code
import numpy as np
from yroots.utils import transform
from scipy.fftpack import fftn
from itertools import product
from time import time

class M_maker:
    def __init__(self,f,a,b,deg,return_inf_norm=False):
        self.f = f
        self.a = a
        self.b = b
        self.deg = deg
        self.return_inf_norm = return_inf_norm

        if self.return_inf_norm == True:
            self.M, self.inf_norm = self.interval_approximate_nd(self.f,self.a,self.b,self.deg,self.return_inf_norm)
            self.M2 = self.interval_approximate_nd(self.f,self.a,self.b,2*self.deg,self.return_inf_norm)[0]
            self.err = np.abs(np.sum(np.abs(self.M)) - np.sum(np.abs(self.M2)))
        else:
            self.M = self.interval_approximate_nd(self.f,self.a,self.b,self.deg)
            self.M2 = self.interval_approximate_nd(self.f,self.a,self.b,2*self.deg)
            self.err = np.abs(np.sum(np.abs(self.M)) - np.sum(np.abs(self.M2)))


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
        x0_slicer = [tuple([slice(None) if i != d else 0 for i in range(dim)])
                    for d in range(dim)]
        deg_slicer = [tuple([slice(None) if i != d else deg for i in range(dim)])
                    for d in range(dim)]
        slices = tuple([slice(0, deg+1)]*dim)
        return x0_slicer, deg_slicer, slices, deg**dim

def norm_pass_or_fail(yroots, roots, tol=2.220446049250313e-13):
    """ Determines whether the roots given pass or fail the test according
        to whether or not their norms are within tol of the norms of the
        "actual" roots, which are determined either by previously known
        roots or Marching Squares roots.
    Parameters
    ----------
        yroots : numpy array
            The roots that yroots found.
        roots : numpy array
            "Actual" roots either obtained analytically or through Marching
            Squares.
        tol : float, optional
            Tolerance that determines how close the roots need to be in order
            to be considered close. Defaults to 1000*eps where eps is machine
            epsilon.

    Returns
    -------
         bool
            Whether or not all the roots were close enough.
    """
    roots_sorted = np.sort(roots,axis=0)
    yroots_sorted = np.sort(yroots,axis=0)
    root_diff = roots_sorted - yroots_sorted
    return np.linalg.norm(root_diff[:,0]) < tol and np.linalg.norm(root_diff[:,1]) < tol

def test_roots_1_1():
    # Test 1.1
        f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
        g = lambda x,y: y-x**6
        f_deg,g_deg = 4,6
        a,b = np.array([-1,-1]),np.array([1,1])
        start = time()
        #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
        f_approx = M_maker(f,a,b,f_deg) #use the class
        g_approx = M_maker(g,a,b,g_deg)
        Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
        yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
        t = time() - start
        actual_roots = np.load('Polished_results/polished_1.1.npy')
        #chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.1.csv', delimiter=',')
        
        return norm_pass_or_fail(yroots,actual_roots),t
        


def test_roots_1_2():
    # Test 1.2
    f = lambda x,y: (y**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((y+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y: ((y+.4)**3-(x-.4)**2)*((y+.3)**3-(x-.3)**2)*((y-.5)**3-(x+.6)**2)*((y+0.3)**3-(2*x-0.8)**3)
    start = time()
    #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    t = time() - start

    # Get Polished results (Newton polishing misses roots)
    yroots2 = solve([f,g],[-1,-1],[1,1], abs_approx_tol=[1e-8, 1e-12], rel_approx_tol=[1e-15, 1e-18],\
                max_cond_num=[1e5, 1e2], good_zeros_factor=[100,100], min_good_zeros_tol=[1e-5, 1e-5],\
                check_eval_error=[True,True], check_eval_freq=[1,2], plot=False, target_tol=[1e-13, 1e-13])
    actual_roots = np.load('Polished_results/polished_1.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, yroots2, 1.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-10)


def test_roots_1_3():
    # Test 1.3
    f = lambda x,y: y**2-x**3
    g = lambda x,y: (y+.1)**3-(x-.1)**2
    f_deg,g_deg = 3,3
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    actual_roots = np.load('Polished_results/polished_1.3.npy')
    #chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.3.csv', delimiter=',')

    return norm_pass_or_fail(yroots,actual_roots),t

def test_roots_1_4():
    # Test 1.4
    f = lambda x,y: x - y + .5
    g = lambda x,y: x + y
    f_deg,g_deg = 1,1
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    a_roots = np.array([[-.25, .25]])
    #chebfun_roots = np.array([np.loadtxt('Chebfun_results/test_roots_1.4.csv', delimiter=',')])

    return norm_pass_or_fail(yroots,a_roots), t

def test_roots_1_5():
    # Test 1.5
    f = lambda x,y: y + x/2 + 1/10
    g = lambda x,y: y - 2.1*x + 2
    f_deg,g_deg = 1,1
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    a_roots = np.array([[0.730769230769231, -0.465384615384615]])

    #chebfun_roots = np.array([np.loadtxt('Chebfun_results/test_roots_1.5.csv', delimiter=',')])

    return norm_pass_or_fail(yroots,a_roots), t

if __name__ == '__main__':
    test_1 = test_roots_1_1()
    test_3 = test_roots_1_3()
    test_4 = test_roots_1_4()
    test_5 = test_roots_1_5()
    print(test_1,test_3,test_4,test_5)
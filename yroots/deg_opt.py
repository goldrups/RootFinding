import numpy as np
from tests.chebfun2_suite import *
from polynomial import MultiPower, MultiCheb
from Combined_Solver import *
import time

max_deg = {1: 100000, 2:1000, 3:9, 4:9, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2}
#randoms. we will not know the max_deg
num_tests = 25
num_funcs = 2


sol_times_2 = []
for i in range(num_tests): #25 tests
    dim = 2 #two dimensions
    a,b = -1*np.ones(dim),np.ones(dim) #assume [-1,1]
    cap_deg = max_deg[dim]*2

    n = np.random.randint(1,cap_deg+1)
    shape = tuple([n]*dim)
    funcs = [MultiPower(np.random.normal(size=shape)) for i in range(num_funcs)]

    start = time.time()
    solve(funcs,a,b)
    end = time.time()
    sol_time = end - start
    sol_times_2.append(sol_time)

sol_times_3 = []
for i in range(num_tests):
    dim = 3 #three dimensions
    a,b = -1*np.ones(dim),np.ones(dim) #assume [-1,1]
    cap_deg = max_deg[dim]*2

    n = np.random.random(1,cap_deg+1)
    shape = tuple([n]*dim)
    funcs = [MultiPower(np.random.normal(size=shape)) for i in range(num_funcs)]

    start = time.time()
    solve(funcs,a,b)
    end = time.time()
    sol_time = end - start
    sol_times_3.append(sol_time)

sol_times_4 = []
#build a function or just re-loop


# possible:
# time both approx and solve
# compare approx and truth over the grid 
# ^(but double or fail err_test method should handle this, right?)


#testsuite polynomials. we will have some where we know the guess_deg. so this is one d. 
#we look for lowest possible guess_deg that gets it.
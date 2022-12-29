import numpy as np
from tests.chebfun2_suite import *
from polynomial import MultiPower, MultiCheb
from Combined_Solver import *
import time

max_deg = {1: 100000, 2:1000, 3:9, 4:9, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2}
#dim 2
dim = 2
#randoms. we will not know the max_deg
cap_deg = max_deg[dim]*2
sol_times = []
for i in range(25): #25 tests
    n = np.random.randint(1,cap_deg+1)
    shape = (n,n)
    funcs = []
    for i in range(dim):
        rand_power_poly = MultiPower(np.random.normal(size=shape))
        funcs.append(rand_power_poly)

    a,b = -1*np.ones(dim),np.ones(dim) #assume [-1,1]
    start = time.time()
    solve(funcs,a,b)
    end = time.time()
    sol_time = end - start
    sol_times.append(sol_time)

    # possible:
    # time both approx and solve
    # compare approx and truth over the grid 
    # ^(but double or fail err_test method should handle this, right?)


#testsuite polynomials. we will have some where we know the guess_deg. so this is one d. 
#we look for lowest possible guess_deg that gets it.
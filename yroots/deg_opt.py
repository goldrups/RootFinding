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
#we look for lowest possible guess_deg that gets it
#TODO: start copy pasting functions and commenting the test cases they come from

#dim 2
dim_2_suite_times = []

test_roots_2_2 = (lambda x,y: x, lambda x,y: (x-.9999)**2 + y**2-1)
test_roots_2_3 = (lambda x,y: np.sin(4*(x + y/10 + np.pi/10)), lambda x,y: np.cos(2*(x-2*y+ np.pi/7)))
test_roots_2_4 = (lambda x,y: np.exp(x-2*x**2-y**2)*np.sin(10*(x+y+x*y**2)), lambda x,y: np.exp(-x+2*y**2+x*y**2)*np.sin(10*(x-y-2*x*y**2)))
test_roots_2_5 = (lambda x,y: 2*y*np.cos(y**2)*np.cos(2*x)-np.cos(y), lambda x,y: 2*np.sin(y**2)*np.sin(2*x)-np.sin(x))
test_roots_4_1 = (lambda x,y: np.sin(3*(x+y)), lambda x,y: np.sin(3*(x-y)))
test_roots_5 = (lambda x,y: 2*x*y*np.cos(y**2)*np.cos(2*x)-np.cos(x*y), lambda x,y: 2*np.sin(x*y**2)*np.sin(3*x*y)-np.sin(x*y))
test_roots_7_3 = (lambda x,y: np.cos(x*y/((1.e-09)**2))+np.sin(3*x*y/((1.e-09)**2)), lambda x,y: np.cos(y/(1.e-09))-np.cos(2*x*y/((1.e-09)**2)))
test_roots_7_4 = (lambda x,y: np.sin(3*np.pi*x)*np.cos(x*y), lambda x,y: np.sin(3*np.pi*y)*np.cos(np.sin(x*y)))
test_roots_8_1 = (lambda x,y: np.sin(10*x-y/10), lambda x,y: np.cos(3*x*y))
test_roots_8_2 = (lambda x,y: np.sin(10*x-y/10) + y, lambda x,y: np.cos(10*y-x/10) - x)
test_roots_9_1 = (lambda x,y: x**2+y**2-.9**2, lambda x,y: np.sin(x*y))
test_roots_10 = (lambda x,y: (x-1)*(np.cos(x*y**2)+2), lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2))

#checks = [check_2_2]

for func in dim_2_suite_funcs:
    start = time.time()
    #approximate both functions in the tuple
    #compare to results
    #if pass, then maybe no need for max_deg, just optimize guess_deg right there....?
    end = time.time()
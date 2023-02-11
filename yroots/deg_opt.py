import numpy as np
from tests.chebfun2_suite import *
from M_maker import M_maker
from polynomial import MultiPower, MultiCheb
from Combined_Solver import *
import time

max_deg = {1: 100000, 2:1000, 3:9, 4:9, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2}
#randoms. we will not know the max_deg
num_tests = 25
#num_funcs = 2


def deg_2():
    sol_times_2 = []
    sol_degs_guess = []
    sol_deg_n = []
    for i in range(num_tests):
        dim = 2
        a,b = -1*np.ones(dim),np.ones(dim) #assume [-1,1]
        cap_deg = max_deg[dim]*2 #double deg

        n = np.random.random(1,cap_deg+1) #now we now it

        shape = tuple([n]*dim)
        fs = [MultiPower(np.random.normal(size=shape)) for i in range(num_funcs)]

        guess_deg = np.random.random(1,n+1)

        a = time.time()
        Ms = [MultiCheb(M_maker(f,a,b,guess_deg)) for f in fs]
        yroots = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision([M.M for M in Ms],[M.err for M in Ms]))
        b = time.time()

        #verify your roots
        #if verify...then append

        sol_times_2.append(b-a)
        sol_degs_guess.append(guess_deg)
        sol_deg_n.append(n)
        
    sol_times_2 = np.array(sol_times_2)
    sol_degs_guess = np.array(guess_deg)
    sol_deg_n = np.array(sol_deg_n)

    data = np.hstack((sol_times_2,sol_degs_guess,sol_deg_n))

    return data

def deg_3():
    sol_times_3 = []
    sol_degs_guess = []
    sol_deg_n = []
    for i in range(num_tests):
        dim = 3
        a,b = -1*np.ones(dim),np.ones(dim) #assume [-1,1]
        cap_deg = max_deg[dim]*2 #double deg

        n = np.random.random(1,cap_deg+1) #now we now it

        shape = tuple([n]*dim)
        fs = [MultiPower(np.random.normal(size=shape)) for i in range(num_funcs)]

        guess_deg = np.random.random(1,n+1)

        a = time.time()
        Ms = [MultiCheb(M_maker(f,a,b,guess_deg)) for f in fs]
        yroots = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision([M.M for M in Ms],[M.err for M in Ms]))
        b = time.time()

        #verify your roots
        #if verify...then append

        sol_times_3.append(b-a)
        sol_degs_guess.append(guess_deg)
        sol_deg_n.append(n)
        
    sol_times_3 = np.array(sol_times_3)
    sol_degs_guess = np.array(guess_deg)
    sol_deg_n = np.array(sol_deg_n)

    data = np.hstack((sol_times_3,sol_degs_guess,sol_deg_n))

    return data


#

#simpler problem
def deg_4():
    sol_times_4 = []
    sol_degs_guess = []
    sol_deg_n = []
    for i in range(num_tests):
        dim = 4
        a,b = -1*np.ones(dim),np.ones(dim) #assume [-1,1]
        cap_deg = max_deg[dim]*2

        n = np.random.random(1,cap_deg+1) #now we now it

        shape = tuple([n]*dim)
        fs = [MultiPower(np.random.normal(size=shape)) for i in range(num_funcs)]

        guess_deg = np.random.random(1,n+1)

        a = time.time()
        Ms = [MultiCheb(M_maker(f,a,b,guess_deg)) for f in fs]
        yroots = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision([M.M for M in Ms],[M.err for M in Ms]))
        b = time.time()

        #verify your roots
        #if verify...then append

        sol_times_4.append(b-a)
        sol_degs_guess.append(guess_deg)
        sol_deg_n.append(n)
        
    sol_times_4 = np.array(sol_times_4)
    sol_degs_guess = np.array(guess_deg)
    sol_deg_n = np.array(sol_deg_n)

    data = np.hstack((sol_times_4,sol_degs_guess,sol_deg_n))

    return data


   
#build a function or just re-loop
#iterate over all dim 4 polynomials, can't just let it know the degree though
#randomly sample max_deg on some uniform dist interval
#randomly sample guess_deg on some (0,max_deg)
#time that shit
#may need to see old versions of the code to make sure we don't tell it the degree


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

# for func in dim_2_suite_funcs:
#     start = time.time()
#     #approximate both functions in the tuple
#     #compare to results
#     #if pass, then maybe no need for max_deg, just optimize guess_deg right there....?
#     end = time.time()
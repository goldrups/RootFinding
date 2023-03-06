import numpy as np
from tests.chebfun2_suite import *
from M_maker import M_maker
from polynomial import MultiPower, MultiCheb
from Combined_Solver import *
import time
from matplotlib import pyplot as plt

max_deg = {1: 100000, 2:1000, 3:9, 4:9, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2}
#randoms. we will not know the max_deg
num_tests = 25
#num_funcs = 2

def get_soln_times(dim,num_tests=25):
    times = []
    guesses = []
    ns = []

    num_funcs = dim

    gran = 5 #granularity
    deg_space = np.arange(1,1+gran)*int(2*max_deg[dim]/gran) #20,40,...
    print(deg_space)

    for n in deg_space:
        sol_times = []
        sol_degs_guess = []
        sol_deg_n = []
        sub_gran = 5
        test_degs = [int(deg) for deg in np.linspace(0,n,sub_gran)][1:]
        for sub_n in test_degs:
            a,b = -1*np.ones(dim),np.ones(dim) #assume [-1,1]
            cap_deg = max_deg[dim]

            #n = np.random.choice(cap_deg+1) #screw that bruh
            #print(n)
            shape = tuple([n]*dim)
            fs = [MultiPower(np.random.normal(size=shape)) for i in range(num_funcs)]

            s = time.time()
            Ms = [M_maker.M_maker(f,a,b,sub_n,dim_deg_cap=[dim,cap_deg]) for f in fs]
            Ms,errs = [MultiCheb(M.M).coeff for M in Ms], [M.err for M in Ms]
            yroots = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision(Ms,errs))
            f = time.time()
            #print("time:", f-s)
            #verify your roots

            #print(type(yroots))
            checks = [soln_check(yroots,MultiCheb(M)) for M in Ms]
            #print(checks)

            if checks == [True]*dim:
                sol_times.append(f-s)
                sol_degs_guess.append(sub_n)
                sol_deg_n.append(n)
            
        times += sol_times
        guesses += sol_degs_guess
        ns += sol_deg_n

    data = np.array([times,guesses,ns])

    return data

def viz(data):
    times = data[0] #solve times
    degs = data[1] #guesses
    true_deg = data[2] #truth

    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlabel("guess")
    ax.set_ylabel("truth")
    ax.set_zlabel("time")

    ax.scatter3D(degs,true_deg,times)
    plt.show()

def soln_check(roots,M):
    """
    Asserts that each root found is indeed a root

    parameters:
    M: polynomial object
    roots: found roots

    returns True if all roots are indeed roots 
    """
    #TODO: perhaps can check double deg if the number of roots increase
    #but i don't think we'll be missing roots due to low guess degree, and certainly not the high cap_deg
    #print(type(M(roots)))
    print(roots)
    match = True
    for root in roots:
        if np.allclose(M(root),np.zeros(len(root))):
            pass
        else:
            match = False
            return match
    return match

   
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

dim_2_suite_tests = [test_roots_2_2, 
                     test_roots_2_3, 
                     test_roots_2_4, 
                     test_roots_2_5, 
                     test_roots_4_1, 
                     test_roots_5, 
                     test_roots_7_3,
                     test_roots_7_4,
                     test_roots_8_1,
                     test_roots_8_2,
                     test_roots_9_1,
                     test_roots_10]

def dim_2_tests():
    for test in dim_2_suite_tests:
        f,g = test
        #increment uniformly to get the optimal degree
        #guess and check back and forth #TODO: next
        #decide on the opt_deg
        #then do the same crap you did with the polynomials
        #

#checks = [check_2_2]

# for func in dim_2_suite_funcs:
#     start = time.time()
#     #approximate both functions in the tuple
#     #compare to results
#     #if pass, then maybe no need for max_deg, just optimize guess_deg right there....?
#     end = time.time()
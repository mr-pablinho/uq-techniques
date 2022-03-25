# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:38:02 2020

@author: PMR
"""


import numpy as np
import chaospy as cp
from scipy.integrate import odeint

# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(w, t, p):
	x1, x2 		= w
	c, k, f, w 	= p

	f = [x2, f*np.cos(w*t) - k*x1 - c*x2]

	return f

# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest):
	sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)

	return sol[20, 0]

if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # the parameters are no longer deterministic
    c_left      = 0.08
    c_right     = 0.12
    k_left      = 0.03
    k_right     = 0.04
    f_left      = 0.08
    f_right     = 0.12
    y0_left     = 0.45
    y0_right    = 0.55
    y1_left     = -0.05
    y1_right    = 0.05

    # w is deterministic
    w = 1.00

    # create uniform distribution object
    distr_c     = cp.Uniform(c_left, c_right)
    distr_k     = cp.Uniform(k_left, k_right)
    distr_f     = cp.Uniform(f_left, f_right)
    distr_y0    = cp.Uniform(y0_left, y0_right)
    distr_y1    = cp.Uniform(y1_left, y1_right)

    # create the multivariate distribution
    distr_5D = cp.J(distr_c, distr_k, distr_f, distr_y0, distr_y1)

    # quad deg 1D
    quad_deg_1D = 3
    poly_deg_1D = 3

    # time domain setup
    t_max       = 20.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])
    t_interest  = len(t)/2

    # create the orthogonal polynomials
    P = cp.orth_ttr(poly_deg_1D, distr_5D)
    

    #################### full grid computations #####################
    # get the non-sparse quadrature nodes and weight
    nodes_full, weights_full = cp.generate_quadrature(quad_deg_1D, distr_5D, rule='G', sparse=False)
    # create vector to save the solution
    sol_odeint_full  = np.zeros(len(nodes_full.T))

    # perform non-sparse pseudo-spectral approximation
    for j, n in enumerate(nodes_full.T):
        # each n is a vector with 5 components
        # n[0] = c, n[1] = k, c[2] = f, n[4] = y0, n[5] = y1
        init_cond               = n[3], n[4]
        args                    = n[0], n[1], n[2], w
        sol_odeint_full[j]      = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)

    # obtain the gpc approximation
    sol_gpc_full_approx = cp.fit_quadrature(P, nodes_full, weights_full, sol_odeint_full)

    # compute statistics
    mean_full    = cp.E(sol_gpc_full_approx, distr_5D)
    var_full     = cp.Var(sol_gpc_full_approx, distr_5D)
    ##################################################################

    #################### sparse grid computations #####################
    # get the sparse quadrature nodes and weight
    nodes_sparse, weights_sparse = cp.generate_quadrature(quad_deg_1D, distr_5D, rule='G', sparse=True)
    # create vector to save the solution
    sol_odeint_sparse  = np.zeros(len(nodes_sparse.T))

    # perform sparse pseudo-spectral approximation
    for j, n in enumerate(nodes_sparse.T):
        # each n is a vector with 5 components
        # n[0] = c, n[1] = k, c[2] = f, n[4] = y0, n[5] = y1
        init_cond               = n[3], n[4]
        args                    = n[0], n[1], n[2], w
        sol_odeint_sparse[j]    = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)

    # obtain the gpc approximation
    sol_gpc_sparse_approx = cp.fit_quadrature(P, nodes_sparse, weights_sparse, sol_odeint_sparse)

    # compute statistics
    mean_sparse    = cp.E(sol_gpc_sparse_approx, distr_5D)
    var_sparse     = cp.Var(sol_gpc_sparse_approx, distr_5D)
    ##################################################################
    print (mean_full.dtype, var_full.dtype)
    print( 'Result:')
    print ("Grid \t| #Points \t| Mean \t\t\t| Var ")
    print ("-----------------------------------------------------------------")
    print ("Full \t|", len(nodes_full.T), "\t\t|", "{a:1.12}".format(a=mean_full), '\t|', "{a:1.12}".format(a=var_full))
    print ("Sparse \t|", len(nodes_sparse.T), "\t\t|", "{a:1.12}".format(a=mean_sparse), '\t|', "{a:1.12}".format(a=var_sparse))
   
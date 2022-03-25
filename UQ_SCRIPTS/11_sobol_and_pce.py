# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:49:17 2020

@author: PMR
"""


import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *

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

	return sol[30, 0]

if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # the parameters are no longer deterministic
    c_left      = 0.08
    c_right     = 0.12
    c_vol       = (c_right - c_left)
    k_left      = 0.03
    k_right     = 0.04
    k_vol       = (k_right - k_left)
    f_left      = 0.08
    f_right     = 0.12
    f_vol       = (f_right - f_left)
    y0_left     = 0.45
    y0_right    = 0.55
    y0_vol      = (y0_right - y0_left)
    y1_left     = -0.05
    y1_right    = 0.05
    y1_vol      = (y1_right - y1_left)

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
    quad_deg_1D = 5
    poly_deg_1D = 2

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

    # perform sparse pseudo-spectral approximation
    for j, n in enumerate(nodes_full.T):
        # each n is a vector with 5 components
        # n[0] = c, n[1] = k, c[2] = f, n[4] = y0, n[5] = y1
        init_cond               = n[3], n[4]
        args                    = n[0], n[1], n[2], w
        sol_odeint_full[j]      = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)

    # obtain the gpc approximation
    sol_gpc_full_approx = cp.fit_quadrature(P, nodes_full, weights_full, sol_odeint_full)

    # compute first order and total Sobol' indices
    first_order_Sobol_ind_full   = cp.Sens_m(sol_gpc_full_approx, distr_5D)
    total_Sobol_ind_full         = cp.Sens_t(sol_gpc_full_approx, distr_5D)
    ##################################################################

    #################### full grid computations #####################
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

    # compute first order and total Sobol' indices
    first_order_Sobol_ind_sparse   = cp.Sens_m(sol_gpc_full_approx, distr_5D)
    total_Sobol_ind_sparse         = cp.Sens_t(sol_gpc_full_approx, distr_5D)
    ##################################################################

    print ('the number of quadrature nodes, i.e. model evaluation, for the full grid is', len(nodes_full.T))
    print ("the first order Sobol' indices are", first_order_Sobol_ind_full  )
    print ("the total Sobol' indices are", total_Sobol_ind_full)

    print ('the number of quadrature nodes, i.e. model evaluation, for the sparse grid is', len(nodes_sparse.T))
    print ("the first order Sobol' indices are", first_order_Sobol_ind_sparse  )
    print ("the total Sobol' indices are", total_Sobol_ind_sparse)


    Sobol_indices_x     = [1, 2, 3, 4, 5]

    labels = ['c', 'k', 'f', r"$y_0$", r"$y_1$"]
    '''
    fs_xticks = 200
    fs_yticks = 150
    fs_title = 220
    fs_legend = 130
    fsize = (60,40)
    ms = 70
    lw = 20
    '''
    fig1 = figure()
    barlist = bar(Sobol_indices_x, total_Sobol_ind_sparse, align='center')
    barlist[0].set_color('r')
    barlist[1].set_color('b')
    barlist[2].set_color('g')
    barlist[3].set_color('m')
    barlist[4].set_color('k')
    xticks(Sobol_indices_x, labels)
    #yticks(fontsize=fs_yticks)
    title("Total Sobol' indices on full grid")
    tight_layout()
    #savefig("sens_1o_full.png")
    

    fig2 = figure()
    barlist = bar(Sobol_indices_x, total_Sobol_ind_full, align='center')
    barlist[0].set_color('r')
    barlist[1].set_color('b')
    barlist[2].set_color('g')
    barlist[3].set_color('m')
    barlist[4].set_color('k')
    xticks(Sobol_indices_x, labels)
    #yticks(fontsize=fs_yticks)
    title("Total Sobol' indices on sparse grid")
    tight_layout()
    #savefig("sens_1o_sparse.png")

    show()




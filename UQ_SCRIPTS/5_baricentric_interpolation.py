# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:08:22 2020

@author: PMR
"""

import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time

# to perform barycentric interpolation, we'll first compute the barycentric weights
def compute_barycentric_weights(grid):
    size    = len(grid)
    w       = np.ones(size)

    for j in range(1, size):
        for k in range(j):
            diff = grid[k] - grid[j]

            w[k] *= diff
            w[j] *= -diff

    for j in range(size):
        w[j] = 1./w[j]

    return w

# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point, grid, weights, func_eval):
    interp_size = len(func_eval)
    L_G         = 1.
    res         = 0.

    for i in range(interp_size):
        L_G   *= (eval_point - grid[i])

    for i in range(interp_size):
        if abs(eval_point - grid[i]) < 1e-10:
            res = func_eval[i]
            L_G    = 1.0
            break
        else:
            res += (weights[i]*func_eval[i])/(eval_point - grid[i])

    res *= L_G 

    return res

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

	return sol[t_interest, 0]

if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.

    # w is no longer deterministic
    w_left      = 1.5
    w_right     = 2.5
    stat_ref    = [-0.43893703, 0.00019678]

    # create uniform distribution object
    distr_w = cp.Uniform(w_left, w_right)

    # no of samples from Monte Carlo sampling
    no_samples_vec = [10, 100, 1000]
    no_grid_points_vec = [5, 10, 20]

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])
    t_interest  = -1
    no_test_points = 200
    test_grid_w         = np.linspace(w_left, w_right, no_test_points)

    # initial conditions setup
    init_cond   = y0, y1

    # create vectors to contain the expectations and variances
    err_exps_mcs = np.zeros(len(no_samples_vec))
    err_vars_mcs = np.zeros(len(no_samples_vec))

    err_exps_lagrange = np.zeros( (len(no_grid_points_vec), len(no_samples_vec)) )
    err_vars_lagrange = np.zeros( (len(no_grid_points_vec), len(no_samples_vec)) )

    # compute relative error
    relative_err = lambda approx, real: abs(1. - approx/real)

    lagrange_time = np.zeros( (len(no_grid_points_vec), len(no_samples_vec)) )
    mc_time = np.zeros(len(no_samples_vec))

    func_eval_test = np.zeros(no_test_points)
    for n, w in enumerate(test_grid_w):
            args                = c, k, f, w
            func_eval_test[n]   = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)

    figure()
    plot(test_grid_w, func_eval_test, 'r:', label='Exact', linewidth=4)

    # perform Monte Carlo sampling
    for j, no_grid_points in enumerate(no_grid_points_vec):
        # generate the unform grid, the function's evaluations and perform the interpolation
        uniform_grid        = np.linspace(w_left, w_right, no_grid_points)
        cheb_grid           = np.array(
                                [0.5*(w_left+w_right) + 
                                 0.5*(w_right-w_left)*np.cos((2*i - 1)/(2.*no_grid_points) * np.pi) 
                                 for i in range(1, no_grid_points+1)]
                                    )
        cur_grid            = cheb_grid

        # evaluate the ode on the grid and compute barycentric weights
        
        start = time.time()
        func_eval = np.zeros(no_grid_points)
        for n, w in enumerate(cur_grid):
                args                = c, k, f, w
                func_eval[n]   = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)
        weights = compute_barycentric_weights(cur_grid)
        end = time.time()
        lagrange_time[j, :] += end-start

        # plot the interpolation function over a test grid
        sol_odeint_lagrange_test = np.array([barycentric_interp(x_, cur_grid, weights, func_eval) for x_ in test_grid_w])
        plot(test_grid_w, sol_odeint_lagrange_test, 'k-', alpha=0.6*((j+1)/2), label='Lagrange: ' + str(no_grid_points))

        for i, no_samples in enumerate(no_samples_vec):
            np.random.seed(1)
            samples_mcs = distr_w.sample(size=no_samples)

            if j == 0:
                start = time.time()
                # Direct Monte Carlo sampling; for each input sample, compute the underlying solution
                sol_odeint_mcs = np.zeros(no_samples)
                for n, w in enumerate(samples_mcs):
                    args                = c, k, f, w
                    sol_odeint_mcs[n]   = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)
                # compute statistics
                err_exps_mcs[i] = relative_err(np.mean(sol_odeint_mcs), stat_ref[0])
                err_vars_mcs[i] = relative_err(np.var(sol_odeint_mcs, ddof=1), stat_ref[1])
                end = time.time()
                mc_time[i] = end-start

            start = time.time()
            # Lagrange interpolation
            sol_odeint_lagrange = np.array([barycentric_interp(x_, cur_grid, weights, func_eval) for x_ in samples_mcs])
            # compute statistics
            err_exps_lagrange[j, i] = relative_err(np.mean(sol_odeint_lagrange), stat_ref[0])
            err_vars_lagrange[j, i] = relative_err(np.var(sol_odeint_lagrange, ddof=1), stat_ref[1])
            end = time.time()
            lagrange_time[j, i] += end-start

    legend()
    show()
    print ("Errors")
    print ("Mean")
    print (err_exps_lagrange)
    print (err_exps_mcs)
    print ("Vars")
    print (err_vars_lagrange)
    print (err_vars_mcs)
    print ("Time")
    print (lagrange_time)
    print (mc_time)
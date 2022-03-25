# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:47:18 2020

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
    ### deterministic setup ###

    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.

    # time domain setup
    t_max       = 20.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])
    t_interest  = len(t)/2

    # initial conditions setup
    init_cond   = y0, y1

    ### stochastic setup ####
    # w is no longer deterministic
    w_left      = 0.95
    w_right     = 1.05

    # create uniform distribution object
    distr_w = cp.Normal(1, 0.1)

    # the truncation order of the polynomial chaos expansion approximation
    N = [3]

    # the quadrature degree of the scheme used to computed the expansion coefficients
    K = N #[2, 4, 6, 8]

    assert(len(N)==len(K))

    # vector to save the statistics
    exp_m = np.zeros(len(N))
    var_m = np.zeros(len(N))

    exp_cp = np.zeros(len(N))
    var_cp = np.zeros(len(N))

    # perform polynomial chaos approximation + the pseudo-spectral
    for h in range(len(N)):
        poly            = cp.orth_ttr(N[h], distr_w, normed=True)
        nodes, weights  = cp.generate_quadrature(K[h], distr_w, rule='G')

        eval_sol    = np.zeros(len(nodes[0]))
        for k_idx, w in enumerate(nodes[0]):
            # w is now a quadrature node
            args        = c, k, f, w
            eval_sol[k_idx] = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)

        # perform polynomial chaos approximation + the pseudo-spectral approach manually
        gpc_coeff = np.zeros(len(poly))
        for i in range(len(poly)):
            for k in range(len(nodes[0])):
                gpc_coeff[i] += eval_sol[k]*poly[i](nodes[0][k])*weights[k]
                print(poly[i])
                print("~~~~~~~~")
        exp_m[h] = gpc_coeff[0]
        var_m[h] = np.sum([gpc_coeff[m]**2 for m in range(1, len(poly))])

         # perform polynomial chaos approximation + the pseudo-spectral approach using chaospy
        gPC_M = cp.fit_quadrature(poly, nodes, weights, eval_sol)

        exp_cp[h] = cp.E(gPC_M, distr_w)
        var_cp[h] = cp.Var(gPC_M, distr_w)

    exp_mc = np.zeros(len(K))
    var_mc = np.zeros(len(K))
    no_samples_vec = [1000]
    # perform Monte Carlo sampling
    for i, no_samples in enumerate(no_samples_vec):
        samples     = distr_w.sample(size=no_samples)
        sol_odeint  = np.zeros( no_samples )

        # Monte Carlo sampling; for each input sample, compute the underlying solution
        for j, w in enumerate(samples):
            args            = c, k, f, w
            sol_odeint[j]   = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)

        # compute statistics
        exp_mc[i] = np.mean(sol_odeint)
        var_mc[i] = np.var(sol_odeint, ddof=1)

    print ('MEAN')
    print ("K | N | Manual \t\t| ChaosPy \t\t| Monte Carlo (Samples)")
    for h in range(len(N)):
        print (K[h], '|', N[h], '|', "{a:1.12f}".format(a=exp_m[h]), '\t|', "{a:1.12f}".format(a=exp_cp[h]), '\t|', "{a:1.12f}".format(a=exp_mc[h]), '(', no_samples_vec[h],')')

    print ('VARIANCE')
    print ("K | N | Manual \t\t| ChaosPy \t\t| Monte Carlo (Samples)")
    for h in range(len(N)):
        print (K[h], '|', N[h], '|', "{a:1.12f}".format(a=var_m[h]), '\t|', "{a:1.12f}".format(a=var_cp[h]), '\t|', "{a:1.12f}".format(a=var_mc[h]), '(', no_samples_vec[h],')')

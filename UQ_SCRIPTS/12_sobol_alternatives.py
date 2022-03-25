# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:24:21 2020

@author: PMR
"""

import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time

# General setup of constants
atol = 1e-10
rtol = 1e-10

c_left, c_right = 0.08, 0.12
c_vol = (c_right - c_left)
k_left, k_right = 0.03, 0.04
k_vol = (k_right - k_left)
f_left, f_right = 0.08, 0.12
f_vol = (f_right - f_left)
y0_left, y0_right = 0.45, 0.55
y0_vol = (y0_right - y0_left)
y1_left, y1_right = -0.05, 0.05
y1_vol = (y1_right - y1_left)

# w is deterministic
w = 1.00

# create uniform distribution objects
distr_c = cp.Uniform(c_left, c_right)
distr_k = cp.Uniform(k_left, k_right)
distr_f = cp.Uniform(f_left, f_right)
distr_y0 = cp.Uniform(y0_left, y0_right)
distr_y1 = cp.Uniform(y1_left, y1_right)

# create the multivariate distribution
distr_5D = cp.J(distr_c, distr_k, distr_f, distr_y0, distr_y1)

# time domain setup
t_max = 10.
dt = 0.01
grid_size = int(t_max / dt) + 1
t = np.array([i * dt for i in range(grid_size)])
t_interest = len(t) - 1

def model(w, t, p):
    x1, x2 = w
    c, k, f, w = p
    f = [x2, f*np.cos(w * t) - k * x1 - c * x2]
    return f


def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest):
    sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)
    return sol[10, 0]


    # relative and absolute tolerances for the ode int solver


def sobol_indices(A, B):
    n, k = A.shape  # n sample size, k number of factors
    s_first = np.zeros(k)
    s_total = np.zeros(k)
    w = 1
    fA, fB = np.zeros(n), np.zeros(n)
    fBA, fAB, = np.zeros((k, n)), np.zeros((k, n))  # k rows, n cols.
    #fBA, fAB = 0.0, 0.0
    # fBA has all factors from B except one and fAB has all from A except one
    for j in range(n):
        init_cond = A[j, 3], A[j, 4]
        args = A[j, 0], A[j, 1], A[j, 2], w
        fA[j] = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)
        init_cond = B[j, 3], B[j, 4]
        args = B[j, 0], B[j, 1], B[j, 2], w
        fB[j] = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)

    # i = 0
    for factor in range(k):
        for j in range(n):
            init_cond = B[j, 3], B[j, 4]
            args = B[j, 0], B[j, 1], B[j, 2], w
            if factor == 0:
                args = A[j, 0], B[j, 1], B[j, 2], w
            elif factor == 1:
                args = B[j, 0], A[j, 1], B[j, 2], w
            elif factor == 2:
                args = B[j, 0], B[j, 1], A[j, 2], w
            elif factor == 3:
                init_cond = A[j, 3], B[j, 4]
            elif factor == 4:
                init_cond = B[j, 3], A[j, 4]
            fBA[factor, j] = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)
            #fBA = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)
            #s_first[factor] += fA[j] * (fBA - fB[j])  # Equation (16) in the paper

            init_cond = A[j, 3], A[j, 4]
            args = A[j, 0], A[j, 1], A[j, 2], w
            if factor == 0:
                args = B[j, 0], A[j, 1], A[j, 2], w
            elif factor == 1:
                args = A[j, 0], B[j, 1], A[j, 2], w
            elif factor == 2:
                args = A[j, 0], A[j, 1], B[j, 2], w
            elif factor == 3:
                init_cond = B[j, 3], A[j, 4]
            elif factor == 4:
                init_cond = A[j, 3], B[j, 4]
            fAB[factor, j] = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)
            #fAB = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)
            #s_total[factor] += (fA[j] - fAB) * (fA[j] - fAB)
        for j in range(n):
            s_first[factor] += fA[j] * (fBA[factor, j] - fB[j])  # Equation (16) in the paper
            s_total[factor] += (fA[j] - fAB[factor, j]) * (fA[j] - fAB[factor, j])
    variance = np.var(np.concatenate((fA, fB)), ddof=1)
    #variance = np.var(np.concatenate((fAB, fBA)), ddof=1)
    s_first = s_first / (n * variance)
    s_total = s_total / (2 * n * variance)
    return s_first, s_total


def get_full_grid_indices(degree):
    P = cp.orth_ttr(degree, distr_5D)
    # get the non-sparse quadrature nodes and weight
    nodes_full, weights_full = cp.generate_quadrature(degree, distr_5D, rule='G', sparse=False)
    # create vector to save the solution
    sol_odeint_full = np.zeros(len(nodes_full.T))
    for j, n in enumerate(nodes_full.T):
        # each n is a vector with 5 components
        # n[0] = c, n[1] = k, c[2] = f, n[4] = y0, n[5] = y1
        init_cond = n[3], n[4]
        args = n[0], n[1], n[2], w
        sol_odeint_full[j] = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)
    sol_gpc_full_approx = cp.fit_quadrature(P, nodes_full, weights_full, sol_odeint_full)

    # compute first order and total Sobol' indices
    first_order_Sobol_ind_full = cp.Sens_m(sol_gpc_full_approx, distr_5D)
    total_Sobol_ind_full = cp.Sens_t(sol_gpc_full_approx, distr_5D)
    return first_order_Sobol_ind_full, total_Sobol_ind_full


def get_sparse_grid_indices(degree):
    P = cp.orth_ttr(degree, distr_5D)
    # get the sparse quadrature nodes and weight
    nodes_sparse, weights_sparse = cp.generate_quadrature(degree, distr_5D, rule='G', sparse=True)
    # create vector to save the solution
    sol_odeint_sparse = np.zeros(len(nodes_sparse.T))

    # perform sparse pseudo-spectral approximation
    for j, n in enumerate(nodes_sparse.T):
        # each n is a vector with 5 components
        # n[0] = c, n[1] = k, c[2] = f, n[4] = y0, n[5] = y1
        init_cond = n[3], n[4]
        args = n[0], n[1], n[2], w
        sol_odeint_sparse[j] = discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest)
    # obtain the gpc approximation
    sol_gpc_sparse_approx = cp.fit_quadrature(P, nodes_sparse, weights_sparse, sol_odeint_sparse)
    # compute first order and total Sobol' indices
    first_order_Sobol_ind_sparse = cp.Sens_m(sol_gpc_sparse_approx, distr_5D)
    total_Sobol_ind_sparse = cp.Sens_t(sol_gpc_sparse_approx, distr_5D)
    return first_order_Sobol_ind_sparse, total_Sobol_ind_sparse


def plot_indices(first_indices, total_indices, title, fname):
    x_labels = ['c', 'k', 'f', r"$y_0$", r"$y_1$"]
    x_array = np.arange(len(x_labels))
    width = 0.35
    fig, ax = subplots()
    ax.bar(x_array - width/2, first_indices, width, label="First order", zorder=3)
    ax.bar(x_array + width/2, total_indices, width, label="Total order", zorder=3)
    ax.set_title(title), ax.set_ylabel("Sobol indices")
    ax.set_xticks(x_array), ax.set_xticklabels(x_labels)
    yticks(np.arange(0., 1., step=0.05))
    ax.grid(zorder=0), ax.legend(), fig.tight_layout()
    #savefig(fname)
    return


def print_indices(indices):
    output = "%2.6f" % indices[0]
    for index in indices[1:]:
        output += " & %2.6f" % index
    output += " // /hline"
    print(output)


degrees = [3, 4]
for degree in degrees:
    print("-----------")
    num_samples = (degree+1)**5  # For a MC sample size equal to number of full grid points
    num_factors = 5
    start = time.time()
    # Matrix setup for Sobol as in paper
    A = np.zeros((num_samples, num_factors))
    A[:, 0] = distr_c.sample(num_samples)
    A[:, 1] = distr_k.sample(num_samples)
    A[:, 2] = distr_f.sample(num_samples)
    A[:, 3] = distr_y0.sample(num_samples)
    A[:, 4] = distr_y1.sample(num_samples)
    B = np.zeros((num_samples, num_factors))
    B[:, 0] = distr_c.sample(num_samples)
    B[:, 1] = distr_k.sample(num_samples)
    B[:, 2] = distr_f.sample(num_samples)
    B[:, 3] = distr_y0.sample(num_samples)
    B[:, 4] = distr_y1.sample(num_samples)

    paper_indices_first, paper_indices_total = sobol_indices(A, B)
    stop = time.time()
    print("Using %d samples. Method from paper got initial sums: %f and %f" %
          (num_samples, np.sum(paper_indices_first), np.sum(paper_indices_total)))
    paper_indices_first = paper_indices_first / np.sum(paper_indices_first)
    paper_indices_total = paper_indices_total / np.sum(paper_indices_total)
    print("First order Sobol indices:")
    print_indices(paper_indices_first)
    print("Total order Sobol indices:")
    print_indices(paper_indices_total)
    print("Sums after further normalization: %f and %f" % (np.sum(paper_indices_first), np.sum(paper_indices_total)))
    print ("Seconds spent calculating indices:", stop-start)
    plot_indices(paper_indices_first, paper_indices_total,
                 "Method from paper. Sample size %d" % num_samples, "paper_%d_sampl" % num_samples)
    del A
    del B

    ### SPARSE GRID COMPUTATION ###
    print("-----------")
    start = time.time()
    PSE_first_sparse, PSE_total_sparse = get_sparse_grid_indices(degree)
    stop = time.time()
    print("Degree %d. Sparse grid. Intial sums: %f and %f"
          % (degree, np.sum(PSE_first_sparse), np.sum(PSE_total_sparse)))
    PSE_first_sparse = PSE_first_sparse / np.sum(PSE_first_sparse)
    PSE_total_sparse = PSE_total_sparse / np.sum(PSE_total_sparse)
    print("First order Sobol indices:")
    print_indices(PSE_first_sparse)
    print("Total order Sobol indices:")
    print_indices(PSE_total_sparse)
    print("Sums after further normalization: %f and %f" % (np.sum(PSE_first_sparse), np.sum(PSE_total_sparse)))
    print ("Seconds spent calculating sparse grid indices:", stop - start)
    plot_indices(PSE_first_sparse, PSE_total_sparse, "Sparse grid indices. Degree %d" % degree, "sparse%d" % degree)

    ### FULL GRID COMPUTATION ###
    print("-----------")
    start = time.time()
    PSE_first_full, PSE_total_full = get_full_grid_indices(degree)
    stop = time.time()
    print("Degree %d. Full grid. Initial sums: %f and %f" % (degree, np.sum(PSE_first_full), np.sum(PSE_total_full)))
    PSE_first_full = PSE_first_full / np.sum(PSE_first_full)
    PSE_total_full = PSE_total_full / np.sum(PSE_total_full)
    print("First order Sobol indices:")
    print_indices(PSE_first_full)
    print("Total order Sobol indices:")
    print_indices(PSE_total_full)
    print("Sums after further normalization: %f and %f" % (np.sum(PSE_first_full), np.sum(PSE_total_full)))
    print( "Seconds spent calculating full grid indices:", stop - start)
    plot_indices(PSE_first_full, PSE_total_full, "Full grid indices. Degree %d" % degree, "full%d" % degree)

show()

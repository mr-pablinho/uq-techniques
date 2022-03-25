# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:00:26 2020

@author: PMR
"""


import numpy as np
from matplotlib.pyplot import *

# Lagrange's cardinal polynomial
def lagrange_poly(eval_point, i, grid):
    result = 1.0

    for j in range(len(grid)):
        if j != i:
            result *= (eval_point - grid[j])/(grid[i] - grid[j])

    return result

# Lagrange interpolation; be aware to take into consideration the evaluation of the 
# interpolating polynomial at the grid points
def lagrange_interp(eval_point, grid, func_eval):
    interp_size = len(func_eval)
    result      = 0.

    for i in range(interp_size):
        if abs(eval_point - grid[i]) < 1e-10:
            result = func_eval[i]
            break
        else: 
            result += lagrange_poly(eval_point, i, grid) * \
                        func_eval[i]

    return result

# to perform barycentric interpolation, we'll first compute the barycentric weights
def compute_barycentric_weights(grid):
    return NotImplementedError

# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point, grid, func_eval):
    return NotImplementedError

if __name__ == '__main__':
    # define test's function
    test_function = lambda x: np.sin(10*x) #1./(1. + 25*x**2) #np.sin(10*x)
    # test_function = lambda x: np.exp(-x)*x + np.sin(x)

    # define uniform interpolation's grid size
    grid_size_vec = [5, 9, 13, 17]

    # define x axis for plotting
    x = np.linspace(-1, 1, 200)

    test_fct = test_function(x)

    # perform interpolation using Lagrange's and barycentric polynomials over different number of nodes
    for grid_size in grid_size_vec:
        # plot test's function
        figure()
        plot(x, test_fct, 'orange', alpha=0.5, label='test function', linewidth=4)

        # generate the unform grid, the function's evaluations and perform the interpolation
        uniform_grid        = np.linspace(-1, 1, grid_size)
        cur_grid            = uniform_grid

        func_eval           = test_function(cur_grid)
        #print func_eval
        Lagrange_interp     = np.array([lagrange_interp(x_, cur_grid, func_eval) for x_ in x])

        # plot the interpolating function
        plot(x, Lagrange_interp, 'r', alpha=0.5, label = 'Lagrange interp. with N = ' + str(grid_size-1), linewidth=2)
        plot(cur_grid, func_eval, 'bx', label = 'Grid points ' + str(grid_size))
        xlim([-1.1, 1.1])
        xlabel('uniform grid', fontsize=10)
        ylabel('test function', fontsize=10)
        grid(True)

        legend(loc='best', fontsize=9)

    show()
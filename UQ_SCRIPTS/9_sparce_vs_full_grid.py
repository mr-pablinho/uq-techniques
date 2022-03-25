# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:34:08 2020

@author: PMR
"""

import numpy as np
import chaospy as cp
from scipy.integrate import odeint

if __name__ == '__main__':
    # define target function
    f = lambda x: x[0]*x[1]*x[2] + np.sin(x[1] + x[3]) - np.exp(-x[4])*x[0] + x[4]**2 - x[0]
    # create the multivariate distribution
    distr_5D = cp.J(cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform())

    # quad deg 1D
    quad_deg_1D_vec = [2, 3, 4]

    # vector to hold number of quad nodes
    quad_full_len       = np.zeros(len(quad_deg_1D_vec), dtype=int)
    quad_sparse_len     = np.zeros(len(quad_deg_1D_vec), dtype=int)
    
    # vector to hold results
    integral_full   = np.zeros(len(quad_deg_1D_vec))
    integral_sparse = np.zeros(len(quad_deg_1D_vec))

    for i, quad_deg_1D in enumerate(quad_deg_1D_vec): 

        # get the non-sparse quadrature nodes and weight
        nodes_full, weights_full        = cp.generate_quadrature(quad_deg_1D, distr_5D, rule='G', sparse=False)
        # get the sparse quadrature nodes and weight
        nodes_sparse, weights_sparse    = cp.generate_quadrature(quad_deg_1D, distr_5D, rule='G', sparse=True)

        quad_full_len[i]    = len(nodes_full.T)
        quad_sparse_len[i]  = len(nodes_sparse.T)

        integral_full[i]   = np.sum([f(n)*w for n, w in zip(nodes_full.T, weights_full)])
        integral_sparse[i] = np.sum([f(n)*w for n, w in zip(nodes_sparse.T, weights_sparse)])

    for i in range(len(quad_deg_1D_vec)):
        print ('number of quadrature points full grid', quad_full_len[i], 'result = ', integral_full[i])
        print ('number of quadrature points sparse grid', quad_sparse_len[i], 'result = ', integral_sparse[i])
           
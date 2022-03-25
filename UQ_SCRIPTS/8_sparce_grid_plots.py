# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:28:10 2020

@author: PMR
"""

import numpy as np
import chaospy as cp

from matplotlib.pyplot import *

if __name__ == '__main__':
    # create the multivariate distribution
    distr_2D = cp.J(cp.Uniform(), cp.Uniform())
    distr_2D_N = cp.J(cp.Normal(), cp.Normal())

    # quad deg 1D
    quad_deg_1D_vec = [2, 3, 4]

    # vector to hold number of quad nodes
    quad_full_len       = np.zeros(len(quad_deg_1D_vec), dtype=int)
    quad_sparse_len     = np.zeros(len(quad_deg_1D_vec), dtype=int)
    
    # vector to hold results
    integral_full   = np.zeros(len(quad_deg_1D_vec))
    integral_sparse = np.zeros(len(quad_deg_1D_vec))

    # for different number of quadrature degree compute full and sparse grid quadratue
    for i, quad_deg_1D in enumerate(quad_deg_1D_vec): 
        nodes_sparse, weights_sparse    =  cp.generate_quadrature(quad_deg_1D, distr_2D, rule='G', sparse=True)
        figure()
        plot(nodes_sparse.T[:,0], nodes_sparse.T[:,1], 'b.', label="Uniform")
        legend()
        nodes_sparse_N, weights_sparse_N    =  cp.generate_quadrature(quad_deg_1D, distr_2D_N, rule='G', sparse=True)
        
        figure()
        plot(nodes_sparse_N.T[:,0], nodes_sparse_N.T[:,1], 'rx', label="Normal")
        legend()
    show()
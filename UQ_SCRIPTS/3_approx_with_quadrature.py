# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:04:34 2020

@author: PMR
"""

import chaospy as cp
import numpy as np

if __name__ == '__main__':
	# define test function
	f = lambda x: np.sum([x**i for i in range(0, 8)])

	# compute the analytical result of int_0^1 f(x)dx
	integral_f = np.sum([1./(i + 1) for i in range(0, 8)])

	# number of nodes and weights; note that for a given M, chaospy generate M + 1 nodes and weights
	N = [0, 1, 2, 3, 4, 5]

	# the computations are performed with respect to the uniform distribution
	distr = cp.Uniform(0,1)
# 	distr = cp.Normal(0.5,0.273)
		
	evals = np.zeros(len(N))
	for i, n in enumerate(N):
		# generate nodes and weights
		nodes, weights = cp.generate_quadrature(n, distr, rule='G')

		evals[i] = np.sum([f(n)*w for n, w in zip(nodes[0], weights)])

	print ('The analytic result is', integral_f)
	for i, ev in enumerate(evals):
		print ('The approximation using', N[i] + 1, 'quadrature points is ', ev)
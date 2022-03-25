# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:25:25 2020

@author: PMR
"""

import chaospy as cp
import numpy as np

if __name__ == '__main__':
	np.set_printoptions(precision=7)
	np.set_printoptions(suppress=True)

	# define the two distributions
	unif_distr = cp.Uniform(-1, 1)
	norm_distr = cp.Normal(mu=10.0, sigma=1)

	# degrees of the polynomials
	N = [3, 4]	

	# generate orthogonal polynomials for all N's
	for i, n in enumerate(N):
		
		# employ the three terms recursion scheme to generate orthonormal polynomials w.r.t. the two distributions
		poly_unif = cp.orth_ttr(n, unif_distr, normed=True)
		poly_norm = cp.orth_ttr(n, norm_distr, normed=True)

		orth_res_unif = np.zeros( (n+1, n+1) )
		orth_res_norm = np.zeros( (n+1, n+1) )
		# compute <\phi_j(x), \phi_k(x)>_\rho
		for j in range(0, n+1):
			for k in range(j, n+1):
				res_unif = cp.E(poly_unif[j]*poly_unif[k], unif_distr)
				res_norm = cp.E(poly_norm[j]*poly_norm[k], norm_distr)
				orth_res_unif[j, k] = res_unif
				orth_res_norm[j, k] = res_norm

		print ("N:", n)
		print ("Uniform:")
		print (orth_res_unif)
		print ("Normal:")
		print (orth_res_norm)
		print ("-----------------------")

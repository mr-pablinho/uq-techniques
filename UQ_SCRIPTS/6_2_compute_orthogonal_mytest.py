# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:55:42 2020

@author: PMR
"""


import chaospy as cp
import numpy as np

np.set_printoptions(precision=7)
np.set_printoptions(suppress=True)

# define the two distributions
unif_distr = cp.Uniform(-1, 1)
norm_distr = cp.Normal(mu=10.0, sigma=1)
join_distr = cp.J(unif_distr, norm_distr)

# degrees of the polynomials
N = 2
M = [12]

orth_res_join = np.zeros(((N+1)+(N+1),(N+1)+(N+1)))
orth_res_unif = np.zeros(((N+1),(N+1)))
orth_res_norm = np.zeros(((N+1),(N+1)))

normed = True
poly_join = cp.orth_ttr(N, join_distr, normed=normed)
poly_unif = cp.orth_ttr(N, unif_distr, normed=normed)
poly_norm = cp.orth_ttr(N, norm_distr, normed=normed)

print(poly_join)
for j in range(0, N+4):
    for k in range(j, N+4):
        res_join = cp.E(poly_join[j]*poly_join[k], join_distr)
        orth_res_join[j,k] = res_join

for j in range(0, N+1):
    for k in range(j, N+1):
        res_unif = cp.E(poly_unif[j]*poly_unif[k], unif_distr)
        res_norm = cp.E(poly_norm[j]*poly_norm[k], norm_distr)
        orth_res_unif[j,k] = res_unif
        orth_res_norm[j,k] = res_norm

print("N:", N)

print("Joint:")
print(orth_res_join)
print("-"*45)
print("Uniform:")
print(orth_res_unif)
print("-"*45)
print("Normal:")
print(orth_res_norm)
print("-"*45)

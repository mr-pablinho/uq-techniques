# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:27:04 2020
TRYING DIFFERENT VALUES FOR N AND K
@author: PMR
"""

# %% Libraries
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt


# %% Function to solve
def foo(coord, a, b):
    return a * np.e ** (-b * coord)


# %% Variables and conditions
coord = np.linspace(0, 20, 200)
dist_a = cp.Uniform(1, 2)
dist_b = cp.Uniform(0.1, 0.2)
# dist_joint = cp.J(dist_a)
dist_joint = cp.J(dist_a, dist_b)


# %% MC solution
numSamples = 50
samples = dist_joint.sample(numSamples)

evals = [foo(coord, sample[0], sample[1]) for sample in samples.T]
# evals = [foo(coord, sample[0], sample[1]) for sample in samples.T]

E_mc = np.mean(evals, 0)
S_mc = np.std(evals, 0)


# %% Point collocation solution
polynomial_expansion = cp.orth_ttr(4, dist_joint)
foo_approx_pc = cp.fit_regression(polynomial_expansion, samples, evals)

E_pc = cp.E(foo_approx_pc, dist_joint)
S_pc = cp.Std(foo_approx_pc, dist_joint)


# %% Pseudo-spectral solution
p = 2
N = 5 # polynomial degree or expansion terms
K = 5 # quadrature nodes 

k = (np.math.factorial(N + p)) / ((np.math.factorial(N)) * (np.math.factorial(p))) - 1
print("k is: " + str(k))

polynomial_expansion = cp.orth_ttr(N, dist_joint)
absissas, weights = cp.generate_quadrature(order=K, dist=dist_joint, rule="clenshaw_curtis")
evals_ps = [foo(coord, val[0], val[1]) for val in absissas.T]
foo_approx_ps = cp.fit_quadrature(polynomial_expansion, absissas, weights, evals_ps)

E_ps = cp.E(foo_approx_ps, dist_joint)
S_ps = cp.Std(foo_approx_ps, dist_joint)

# %% Plot solutions

# MC solution
title_fig1 = "MC solution - Samples: " + str(numSamples)
plt.figure(title_fig1)
for i in range(numSamples):
    plt.plot(coord, evals[i], color='blue', alpha=0.20, linewidth=2)
plt.title(title_fig1)

title_fig2 = "MC solution - mean and standard deviation"
plt.figure(title_fig2)
plt.plot(coord, E_mc, color='black', linewidth=2)
plt.fill_between(coord, E_mc - (2*S_mc), E_mc + (2*S_mc), color="darkslateblue", linewidth=0)
plt.fill_between(coord, E_mc - (S_mc), E_mc + (S_mc), color="slateblue", linewidth=0)
plt.title(title_fig2)

# PC solution
title_fig3 = "PC solution - mean and standard deviation"
plt.figure(title_fig3)
plt.plot(coord, E_pc, color='black', linewidth=2)
plt.fill_between(coord, E_pc - (2*S_pc), E_pc + (2*S_pc), color="darkslateblue", linewidth=0)
plt.fill_between(coord, E_pc - (S_pc), E_pc + (S_pc), color="slateblue", linewidth=0)
plt.title(title_fig3)

# PS solution
title_fig4 = "PS solution - mean and standard deviation (N=" + str(N) +", K="+ str(K) + ")"
plt.figure(title_fig4)
plt.plot(coord, E_ps, color='black', linewidth=2)
plt.fill_between(coord, E_ps - (2*S_ps), E_ps + (2*S_ps), color="darkslateblue", linewidth=0)
plt.fill_between(coord, E_ps - (S_ps), E_ps + (S_ps), color="slateblue", linewidth=0)
plt.title(title_fig4)



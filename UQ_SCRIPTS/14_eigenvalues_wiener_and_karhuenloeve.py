# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:37:07 2020

@author: PMR
"""


import numpy as np
import matplotlib.pyplot as plt

t_min, t_max = 0., 1.
N = 1000
M = [10, 10000]
dt = float(t_max - t_min) / N
TIME = [t * dt for t in range(N)]

# Computing and plotting eigenvalues
eigvals = np.zeros(shape=max(M)-1) # eigvals[n] corresponds to lambda_{n+1}. e.g. eigvals[0] = lambda_1
for n in range(1, max(M)):
    eigvals[n - 1] = 1/((n - 0.5)*(n - 0.5) * np.pi * np.pi)
plt.semilogy(range(1, max(M)), eigvals)
plt.title("Eigenvalues"), plt.xlabel("n")
plt.xticks(np.arange(0, max(M)+1, max(M)/10)), plt.ylabel(r"$\lambda_n$")
plt.grid(True, which='both'), plt.grid(True, which='minor', alpha=0.8)
#plt.savefig("eigenvals")

# Calculating and plotting a Wiener process
dW = np.random.normal(0, np.sqrt(dt), len(TIME))
dW[0] = 0
W = np.cumsum(dW)
plt.figure()
plt.plot(TIME, W), plt.title(r"A Wiener process for $t \in [0,1]$ with N = %d" % N)
plt.xlabel("t"), plt.xticks(np.arange(0, t_max + dt, t_max/10)), plt.ylabel(r"$W_t$")
plt.grid(True, which='both')
plt.grid(True, which='minor', alpha=0.9)
#plt.savefig("Wiener_proc")

# Making at plotting KL expansions
zeta = np.random.normal(0, 1, max(M))
kl_exp = np.zeros(shape=len(TIME))
plt.figure()
for m in M:
    for i in range(1, len(kl_exp)):
        w = 0.
        for j in range(1, m):
            w += (np.sin((j - 0.5)*np.pi*i*dt) * zeta[j])/((j - 0.5)*np.pi)
        kl_exp[i] = w * np.sqrt(2)
    plt.plot(TIME, kl_exp, '-.', label="M = %d" % m)
plt.title("Karhunen-Loeve expansion for different M")
plt.xlabel("t"), plt.xticks(np.arange(0, t_max + dt, t_max/10)), plt.ylabel(r"$W_t$")
plt.grid(True, which='both'), plt.grid(True, which='minor', alpha=0.8)
plt.legend()
#plt.savefig("KL_expansions")
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:43:57 2020

@author: PMR
"""


import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

c, k, y0, y1, w, mu = .5, 2., .5, 0., 1., .5
dt = 0.01
T = 10.
N = [10, 100]
M = [5, 10]
t = [t * dt for t in range(int(T/dt))]
plt_colors = ['c', 'm', 'r', 'g']


eigvals = np.zeros(shape=max(M)-1)  # eigvals[n] corresponds to lambda_{n+1}. e.g. eigvals[0] = lambda_1
eigvals_sqrt = np.zeros(shape=max(M)-1)  # Same indexing as above
eigvecs = np.zeros(shape=(max(M)-1, len(t)))  # Same indexing as above, plus additional time dimension

two_div_T = np.sqrt(2.0/T)
for n in range(1, max(M)):
    eigvals[n-1] = T * T / ((n - 0.5) * (n - 0.5) * np.pi * np.pi)
    eigvals_sqrt[n-1] = T / ((n - 0.5) * np.pi)
    for i in range(len(t)):
        eigvecs[n-1, i] = two_div_T * np.sin(((float(n) - 0.5) * np.pi * float(i) * dt) / T)


def discretize_oscillator_sys(f, proc):  # Process argument for plotting
    # initalize the solution vector with zeros
    z0 = np.zeros(shape=len(t))
    z1 = np.zeros(shape=len(t))

    z0[0] = y0
    z1[0] = y1

    # implement the obtained Euler scheme
    for i in range(0, len(t) - 1):
        z1[i + 1] = z1[i] + dt*(-k*z0[i] - c*z1[i] + f[i]*np.cos(w*t[i]))
        z0[i + 1] = z0[i] + dt*z1[i]
    '''
    plt.figure(1)
    if proc == 0:
        lab = "Wiener process"
    else:
        lab = "KL with m = %d" % (M[proc-1])
    plt.plot(t, z0, plt_colors[proc], alpha=0.4, label=lab)
    '''
    return z0[-1]


for n in N:
    solutions = np.zeros(shape=(len(M) + 1, n))
    wiener_means = np.zeros(shape=n)
    # Using Wiener process
    for solu in range(n):
        # Initialize f
        dF = np.random.normal(0, np.sqrt(dt), int(T/dt))
        dF[0] = mu
        f = np.cumsum(dF)
        solutions[0, solu] = discretize_oscillator_sys(f, 0)
        wiener_means[solu] = np.mean(f)
        # Debug plotting
        #plt.figure(0)
        #plt.plot(t, f, 'c', alpha=0.9, label="Wiener proc")
    #plt.title("f values over time using N = %d" % n)
    print("Means of all values of f using Wiener process when n = %d: %1.5f" % (n, np.mean(wiener_means)))
    print("For N = %d:" % n)
    print("\t\t\t\t\tE[y(10)]")
    print("Wiener process:\t\t%f" % np.mean(solutions[0, :]))


    # Using KL expansion
    for index, m in enumerate(M):
        for solu in range(n):
            f = np.zeros(shape=int(T/dt))
            f[0] = mu
            for i in range(1, len(f)):  # Determining f[i]
                f_tmp = 0.
                zeta = np.random.normal(0, 1, m)  # Draw m new random variables for KL exp. step
                for j in range(m-1):  # Here: j corresponds to index n in the exercise sheet
                    f_tmp += eigvals_sqrt[j] * eigvecs[j, i] * zeta[j]
                f[i] = f_tmp + mu
            
            # Plotting for debugging
            plt.figure(0)
            plt.plot(t, f, alpha=0.2, label="KL m = %d" % m)
            
            solutions[index + 1, solu] = discretize_oscillator_sys(f, index + 1)
        print("KL with M = %3d:\t%f" % (m, np.mean(solutions[index + 1, :])))

    
    # Debug plotting
    plt.figure(0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    #plt.savefig("f_values")
    plt.figure(1)
    plt.title("Oscillator trajectories for n = %d" %n)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    #plt.savefig("trajectories")
    plt.show()
    
    print("------------------------------")
    print("\t\t\t\t\tVar[y(10)]")
    print("Wiener process:\t\t%f" % np.var(solutions[0, :], ddof=1))
    for index, m in enumerate(M):
        print("KL with M = %3d:\t%f" % (m, np.var(solutions[index + 1, :], ddof=1)))
    print("\n------------------------------\n")

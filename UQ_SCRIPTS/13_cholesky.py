# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:34:51 2020

@author: PMR
"""

# from __builtin__ import xrange
import matplotlib.pyplot as plt
import numpy as np


N = 35
#l = 1.0
num_plots = 3


def c1(x, y):  # x, y tuples (x,y)
    d = (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1])
    return np.exp(-(np.sqrt(d)))


def c2(x, y):
    d = (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1])
    return np.exp(-d*0.5)


x_min, x_max, y_min, y_max = 0., 1., 0., 1.
mesh_size_x, mesh_size_y = N, N
x_coord = np.arange(x_min + x_max/(2*mesh_size_x), x_max, (x_max - x_min)/mesh_size_x)
y_coord = np.arange(y_min + y_max/(2*mesh_size_y), y_max, (y_max - y_min)/mesh_size_y)

mesh_coord = {}
for i in xrange(mesh_size_x):
    for j in xrange(mesh_size_y):
        mesh_coord[i*mesh_size_x + j] = x_coord[i], y_coord[j]

mesh_coord = mesh_coord.values()  # 100 tuples of (x,y) starting at (0.05, 0.05) going up (y->1) then right (x->1)
C1 = np.zeros(shape=(N*N, N*N))
C2 = np.zeros(shape=(N*N, N*N))
m = np.zeros(shape=N*N)
for i in xrange(N*N):
    for j in xrange(N*N):
        C1[i, j] = c1(mesh_coord[i], mesh_coord[j])
        C2[i, j] = c2(mesh_coord[i], mesh_coord[j])
    C2[i, i] += 1E-12
    m[i] = 0.1

L1 = np.linalg.cholesky(C1)
L2 = np.linalg.cholesky(C2)

zero_vec = np.zeros(shape=N*N)
identity = np.eye(N=N*N)

for plot in range(num_plots):
    dist = np.random.multivariate_normal(zero_vec, identity)
    tmp_mat = np.matmul(L1, dist)
    G1 = m + np.matmul(L1, dist)
    data1 = np.zeros(shape=(N, N))
    G2 = m + np.matmul(L2, dist)
    data2 = np.zeros(shape=(N, N))
    for i in xrange(N):
        for j in xrange(N):
            data1[i, j] = G1[i * mesh_size_x + j]
            data2[i, j] = G2[i * mesh_size_x + j]

    plt.figure()
    plt.title("Plot %d out of %d using first covariance" % (plot+1, num_plots))
    color_map = plt.imshow(data1)
    color_map.set_cmap("Blues_r")
    plt.colorbar()
    plt.savefig("cov1_%d" % (plot + 1))
    plt.figure()
    plt.title("Plot %d out of %d using second covariance" % (plot+1, num_plots))
    color_map = plt.imshow(data2)
    color_map.set_cmap("Blues_r")
    plt.colorbar()
    #plt.savefig("cov2_%d" % (plot + 1))

plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:07:10 2020

@author: PMR
"""


import numpy as np
from matplotlib.pyplot import *

def lagrange_elem(x, a, b):
	return (x-a)/(b-a)

def lagrange_poly(x, xx, j):
	interp = 1.
	for m in range(len(xx)):
		if m is not j:
			interp *= lagrange_elem(x, xx[m], xx[j])
	return interp

if __name__ == '__main__':
	N = [5, 10, 15, 20]

	xxx = np.linspace(-1, 1, 1000)

	print( "Uniform")
	figure()
	for n in range(N[1]):
		xx = np.linspace(-1, 1, N[1])
		yyy = np.zeros(1000)
		for idx, x in enumerate(xxx):
			yyy[idx] = lagrange_poly(x, xx, n)
		plot(xxx, yyy)
	for n in N:
		xx = np.linspace(-1, 1, n)
		lebesgue_constant = -1
		for x in xxx:
			cur_max = 0.
			for j in range(0, n):
				cur_max += abs(lagrange_poly(x, xx, j))

			if cur_max > lebesgue_constant:
				lebesgue_constant = cur_max

		print( n, lebesgue_constant)
	title("Uniform")

	print ("Cheb")
	figure()
	for n in range(N[1]):
		xx = np.array([np.cos((2*i - 1)/(2.*N[1]) * np.pi) for i in range(1, N[1]+1)])
		yyy = np.zeros(1000)
		for idx, x in enumerate(xxx):
			yyy[idx] = lagrange_poly(x, xx, n)
		plot(xxx, yyy)
	for n in N:
		xx = np.array([np.cos((2*i - 1)/(2.*n) * np.pi) for i in range(1, n+1)])

		lebesgue_constant = -1
		for x in xxx:
			cur_max = 0.
			for j in range(0, n):
				cur_max += abs(lagrange_poly(x, xx, j))

			if cur_max > lebesgue_constant:
				lebesgue_constant = cur_max

		print( n, lebesgue_constant)
	title("Cheb")
	show()
	'''
	interp = np.zeros(100)
	figure()
	for j in range(0, n):
		for idx, x in enumerate(xxx):
			interp[idx] = lagrange_poly(x, xx, j)
		plot(xxx, interp)

	show()
	'''
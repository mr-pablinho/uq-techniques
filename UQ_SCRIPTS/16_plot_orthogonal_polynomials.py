# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:58:24 2020

@author: PMR
"""
import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
from mpmath import mp
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application


#%% Legendre polynomials

distribution = cp.Uniform(-1, 1)
grade = [0,1,2,3,4,5]
poly = cp.orth_ttr(grade[-1], distribution, cross_truncation=10)
# poly = poly/poly(1)  # activate: probabilist polynomials / deactivate: physicist polynomials

legend_grade = []

for i in grade: 
    
    poly_string = str(poly[i])
    print("*** legendre polynomial grade " + str(i))
    print(poly_string)
    
    f = lambda q0 : eval(poly_string)
    
    val_a = -1.2
    val_b = 1.2
    delta = 0.001
    steps = int(((val_b - val_a) / delta) + 1)
    x = np.linspace(val_a, val_b, steps)
    legend_grade.append("N = " + str(i))
    
    y = []
    for j in x:
        calc = f(j)
        y.append(calc)
    
    y = np.array(y)
    y[y<=val_a]= np.nan
    y[y>=val_b]= np.nan
    
    plt.figure("Lengendre polynomials")
    y = np.array(y)        
    plt.plot(x, y)
    plt.ylim((-1.2, 1.2))
    plt.xlim((-1.2, 1.2))
    plt.legend(legend_grade)
    plt.title("Lengendre polynomials")

# %% Hermite Polynomials

distribution = cp.Normal(0,1)
grade = [0,1,2,3,4,5]
poly = cp.orth_ttr(grade[-1], distribution, cross_truncation=10)

legend_grade = []

for i in grade:
    
    poly_string = str(poly[i])
    print(poly_string)
    print("*** hermite polynomial grade " + str(i))
    
    f = lambda q0 : eval(poly_string)
    
    val_a = -3
    val_b = 3
    delta = 0.001
    steps = int(((val_b - val_a) / delta) + 1)
    x = np.linspace(val_a, val_b, steps)
    legend_grade.append("N = " + str(i))
    
    y = []
    for j in x:
        calc = f(j)
        y.append(calc)
    
    y = np.array(y)
    y[y<=-8.0]= np.nan
    y[y>=8.0]= np.nan
    
    plt.figure("Hermite polynomials") 
    plt.plot(x, y)
    plt.ylim((-8.0, 8.0))
    plt.xlim((-3.0, 3.0))
    plt.legend(legend_grade)
    plt.title("Hermite polynomials")
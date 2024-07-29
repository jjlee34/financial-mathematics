#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 01:38:12 2024

@author: jordanlee
"""
import math
import itertools
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#symmetric random walk

M = 10 #simulations
t = 10 #time

random_walk = [-1, 1]
steps = np.random.choice(random_walk, size=(M,t)).T
origin = np.zeros((1,M))
rw_paths = np.concatenate([origin, steps]).cumsum(axis=0)

plt.plot(rw_paths)
plt.xlabel("Days (t)")
plt.ylabel("Move")
plt.show()

#quadratic variation

quadratic_variation = lambda x: round(np.square(x[:-1]-x[1:]).sum(),3)
variance = lambda x: round(np.var(x,axis=0),3)

print([quadratic_variation(path) for path in rw_paths.T[:4]])
print([variance(path) for path in rw_paths[1:11]])

#scaled symmetric random walk

M = 10 #simulations
t = 10 #time
n = 10

random_walk = [-1, 1]
steps = (1/np.sqrt(n)) * np.random.choice(random_walk, size=(M,t*n)).T
origin = np.zeros((1,M))
srw_paths = np.concatenate([origin,steps]).cumsum(axis = 0)

time = np.linspace(0,t,t*n+1)
tt = np.full(shape=(M,t*n+1), fill_value = time)
tt = tt.T

plt.plot(tt, srw_paths)
plt.xlabel("Days (t)")
plt.ylabel("Move")
plt.show()

print([quadratic_variation(path) for path in srw_paths.T[:4]])
print([variance(path) for path in srw_paths[1:11]])

#limit of binomial distribution

n = 100 #
t = 10 #time

def nCr(n,k):
    f = math.factorial
    return f(n) / (f(k) * f(n-k))

perms = [nCr(n*t,k)*(0.5)**(n*t) for k in range(int(n*t)+1)]

W_nt = lambda n,t: 1/np.sqrt(n) * np.arange(-n*t,n*t+1,2)

outcomes = W_nt(n,t)
plt.bar(outcomes,[perm/(outcomes[1]-outcomes[0]) for perm in perms],outcomes[1]-outcomes[0],
        label='{0} scaled RW'.format(n))

x = np.linspace(-3*np.sqrt(t), 3*np.sqrt(t), 100)
plt.plot(x, stats.norm.pdf(x, 0, np.sqrt(t)), 'k-',label='normal dist')

plt.xlim(-3*np.sqrt(t), 3*np.sqrt(t))
plt.ylabel("Probability")
plt.xlabel("Move")
plt.legend()
plt.show()

#brownian motion

M = 10 #simulations
t = 10 #time
n = 100 #steps
dt = t/n #time step

steps = np.random.normal(0, np.sqrt(dt), size=(M, n)).T
origin = np.zeros((1,M))
bm_paths = np.concatenate([origin, steps]).cumsum(axis=0)

time = np.linspace(0,t,n+1)
tt = np.full(shape=(M, n+1), fill_value=time)
tt = tt.T

plt.plot(tt,bm_paths)
plt.xlabel("Days (t)")
plt.ylabel("Move")
plt.show()

print([quadratic_variation(path) for path in bm_paths.T[:4]])
print([variance(path) for path in bm_paths[1:11]])
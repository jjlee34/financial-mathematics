#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 01:45:00 2024

@author: jordanlee
"""

import numpy as np
import matplotlib.pyplot as plt

mu = 0.1 # drift coefficient
n = 100 # steps
T = 1 # time
M = 100 # number of simulations
S0 = 100 # initial stock price
sigma = 0.3 # volatility
dt = T/n # time step

St = np.exp(
    (mu - sigma ** 2 / 2)*dt
    + sigma*np.random.normal(0, np.sqrt(dt), size=(M,n)).T
    )

St = np.vstack([np.ones(M), St])
St = S0 * St.cumprod(axis=0)

time = np.linspace(0,T,n+1)
tt = np.full(shape=(M, n+1), fill_value = time).T

plt.plot(tt, St)
plt.xlabel("Years $(t)$")
plt.ylabel("Stock Price $(S_t)$")
plt.title("Realizations of Geomtric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t DW_t$\n $S_0 = {100}, \mu = {0.1}, \sigma = {0.3}$"
          )
plt.show()
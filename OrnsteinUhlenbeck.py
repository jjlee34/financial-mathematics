#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 08:21:00 2024

@author: jordanlee
"""

import time
import math
import numpy as np
import pandas as pd
import datetime
import scipy as sc
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
from IPython.display import display, Latex
from statsmodels.graphics.tsaplots import plot_acf

yf.pdr_override()

def get_data(stocks, start, end):
    df = pdr.get_data_yahoo(stocks, start, end)
    return df

endDate = datetime.datetime.now()
startDate = endDate - datetime.timedelta(days=7000)

stock_prices = get_data('^GSPC', startDate, endDate)

# Log-returns

log_returns = np.log(stock_prices.Close/stock_prices.Close.shift(1)).dropna()
log_returns_sq = np.square(log_returns)

# Rolling volatility

TRADING_DAYS = 40
volatility = log_returns.rolling(window = TRADING_DAYS).std()*np.sqrt(252)
volatility = volatility.dropna()

# Maximum Log-likelihood Estimation (MLE)

def MLE_norm(x):
    mu_hat = np.mean(x)
    sigma2_hat = np.var(x)
    return mu_hat, sigma2_hat

mu = 5
sigma = 2.5
N = 10000

np.random.seed(0)
x = np.random.normal(loc=mu, scale=sigma, size=(N,))
mu_hat, sigma2_hat = MLE_norm(x)

for_mu_hat = '$\hat{\mu} = '+format(round(mu_hat,2))+'$'
for_sigma2_hat = '$\hat{\sigma} = '+format(round(np.sqrt(sigma2_hat),2))+'$'

#display(Latex(for_mu_hat))
#display(Latex(for_sigma2_hat))

# Numerical MLE

def log_likelihood(theta, x):
    mu = theta[0]
    sigma = theta[1]
    l_theta = np.sum(np.log(sc.stats.norm.pdf(x, loc=mu, scale=sigma)))
    return -l_theta

def sigma_pos(theta):
    sigma = theta[1]
    return sigma

cons_set = {'type': 'ineq', 'fun': sigma_pos}
theta0 = [2,3]
opt = sc.optimize.minimize(fun=log_likelihood, x0=theta0, args=(x,), constraints = cons_set)

for_mu_hat = '$\hat{\mu} = '+format(round(opt.x[0],2))+'$'
for_sigma2_hat = '$\hat{\sigma} = '+format(round(opt.x[1],2))+'$'

#display(Latex(for_mu_hat))
#display(Latex(for_sigma2_hat))

# MLE of Ornstein-Uhlenbeck

def mu(x, dt, kappa, theta):
    ekt = np.exp(-kappa*dt)
    return x*ekt + theta*(1-ekt)

def std(dt, kappa, sigma):
    e2kt = np.exp(-2*kappa*dt)
    return sigma*np.sqrt((1-e2kt)/(2*kappa))

def log_likelihood_OU(theta_hat, x):
    kappa = theta_hat[0]
    theta = theta_hat[1]
    sigma = theta_hat[2]
    
    x_dt = x[1:]
    x_t = x[:-1]
    
    dt = 1/252
    
    mu_OU = mu(x_t, dt, kappa, theta)
    sigma_OU = std(dt, kappa, sigma)
    
    l_theta_hat = np.sum(np.log(sc.stats.norm.pdf(x_dt, loc=mu_OU, scale=sigma_OU)))
    
    return -l_theta_hat

def kappa_pos(theta_hat):
    kappa = theta_hat[0]
    return kappa

def sigma_pos(theta_hat):
    sigma = theta_hat[2]
    return sigma

vol = np.array(volatility)

cons_set = [{'type': 'ineq', 'fun': kappa_pos},
            {'type': 'ineq', 'fun': sigma_pos}]
theta0 = [1,1,1]
opt = sc.optimize.minimize(fun=log_likelihood_OU, x0=theta0, args=(vol,), constraints=cons_set)

kappa = round(opt.x[0],3)
theta = round(opt.x[1],3)
sigma = round(opt.x[2],3)
vol0 = vol[-1]

for_kappa_hat = '$\hat{\kappa} = '+str(kappa)+'$'
for_theta_hat = '$\hat{\Theta} = '+str(theta)+'$'
for_sigma_hat = '$\hat{\sigma} = '+str(sigma)+'$'

print('The MLE for data is:')
display(Latex(for_kappa_hat))
display(Latex(for_theta_hat))
display(Latex(for_sigma_hat))
print('Last Volatility', round(vol0,3))

# Simulation of Ornstein-Uhlenbeck

Time = 0.2
M = 10000

Z = np.random.normal(size=(M))

drift_OU = mu(vol0, Time, kappa, theta)
diffusion_OU = std(Time, kappa, sigma)
vol_OU = drift_OU + diffusion_OU*Z

plt.hist(vol_OU)
plt.title('Ornstein-Uhlenbeck Continuous Distribution @ Time')
plt.xlabel('Volatility')
plt.show()

# Discretised SDE

days = 1
years = 2

dt = days/252

M = 1000
N = int(years/dt)

vol_OU = np.full(shape=(N, M), fill_value=vol0)
Z = np.random.normal(size=(N, M))

start_time = time.time()
for t in range(1,N):
    drift_OU = kappa*(theta - vol_OU[t-1])*dt
    diffusion_OU = sigma*np.sqrt(dt)
    vol_OU[t] = vol_OU[t-1] + drift_OU + diffusion_OU*Z[t]
print('Execution time', time.time() - start_time)

vol_OU = np.concatenate( (np.full(shape=(1, M), fill_value=vol0), vol_OU ) )
plt.plot(vol_OU)
plt.title('Ornstein-Uhlenbeck Euler Discretization')
plt.ylabel('Volatility')
plt.show()
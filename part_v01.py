# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:41:55 2024

@author: ieron
"""
import numpy as np
from math import exp, sqrt

# Given values
r = 0.03  # risk-free rate
q = 0.02  # dividend yield
sigma_SX5E = 0.15  # volatility for both SX5E and AEX
sigma_AEX = 0.15
T = 5  # maturity in years
N_SX5E = 10000  # number of contracts for SX5E
N_AEX = 55000  # number of contracts for AEX
S_SX5E_0 = 4235  # initial price for SX5E
S_AEX_0 = 770  # initial price for AEX
K_SX5E = 4235  # strike price for SX5E
K_AEX = 770  # strike price for AEX

# Generating Z_SX5E and Z_AEX from a standard normal distribution
Z_SX5E = np.random.randn()
Z_AEX = np.random.randn()

# Expected price at maturity for SX5E and AEX
S_SX5E_T = S_SX5E_0 * exp((r - q - 0.5 * sigma_SX5E**2) * T + sigma_SX5E * Z_SX5E * sqrt(T))
S_AEX_T = S_AEX_0 * exp((r - q - 0.5 * sigma_AEX**2) * T + sigma_AEX * Z_AEX * sqrt(T))

S_SX5E_T, S_AEX_T

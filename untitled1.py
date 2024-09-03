# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:55:46 2024

@author: ieron
"""

# Re-import necessary libraries and re-define variables after reset
import numpy as np
from math import exp, sqrt
import matplotlib.pyplot as plt

# Given values
r = 0.03  # risk-free rate
q = 0.02  # dividend yield
sigma_SX5E = 0.15  # volatility for SX5E
sigma_AEX = 0.15  # volatility for AEX
T = 5  # maturity in years
S_SX5E_0 = 4235  # initial price for SX5E
S_AEX_0 = 770  # initial price for AEX

# Number of simulations
num_simulations = 10000

# Arrays to store simulation results
simulated_S_SX5E_T = np.zeros(num_simulations)
simulated_S_AEX_T = np.zeros(num_simulations)

# Perform simulations
for i in range(num_simulations):
    Z_SX5E_sim = np.random.randn()
    Z_AEX_sim = np.random.randn()
    
    simulated_S_SX5E_T[i] = S_SX5E_0 * np.exp((r - q - 0.5 * sigma_SX5E**2) * T + sigma_SX5E * Z_SX5E_sim * np.sqrt(T))
    simulated_S_AEX_T[i] = S_AEX_0 * np.exp((r - q - 0.5 * sigma_AEX**2) * T + sigma_AEX * Z_AEX_sim * np.sqrt(T))

# Calculate mean of simulated future prices
mean_S_SX5E_T = np.mean(simulated_S_SX5E_T)
mean_S_AEX_T = np.mean(simulated_S_AEX_T)

mean_S_SX5E_T, mean_S_AEX_T

###############################################################################
# Define the number of paths to simulate and plot
num_paths = 10000

# Create a figure for plotting
plt.figure(figsize=(14, 7))

# Time array for plotting
time_array = np.linspace(0, T, total_steps)

# Simulate multiple paths
for _ in range(num_paths):
    path_S_SX5E = np.zeros(total_steps)
    path_S_AEX = np.zeros(total_steps)
    path_S_SX5E[0] = S_SX5E_0
    path_S_AEX[0] = S_AEX_0

    for t in range(1, total_steps):
        Z_SX5E = np.random.randn()
        Z_AEX = np.random.randn()

        path_S_SX5E[t] = path_S_SX5E[t-1] * np.exp((r - q - 0.5 * sigma_SX5E**2) * dt + sigma_SX5E * Z_SX5E * np.sqrt(dt))
        path_S_AEX[t] = path_S_AEX[t-1] * np.exp((r - q - 0.5 * sigma_AEX**2) * dt + sigma_AEX * Z_AEX * np.sqrt(dt))

    # Plot each path
    plt.plot(time_array, path_S_SX5E, alpha=0.7)
    plt.plot(time_array, path_S_AEX, alpha=0.7)

plt.title('Simulated Paths for SX5E and AEX Equity Prices Over 5 Years')
plt.xlabel('Years')
plt.ylabel('Equity Price')
plt.legend(['SX5E Paths', 'AEX Paths'], loc='upper left')
plt.show()
























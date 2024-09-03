# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:20:39 2024

@author: ieron
"""

import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt

# Given parameters
LGD = 0.4 
q = 0.02
r = 0.03
sigma_sx = 0.15
sigma_aex = 0.15
rho = 0.8
dt = 1/12  # monthly time steps
T = 5  # maturity in years
steps = int(T * 12)  # number of steps
N_simulations = 10000  # number of simulations

# Initial values
S_sx0 = 4235
S_aex0 = 770
K_sx = 4235
K_aex = 770

# Generate correlated random numbers for Z_sx and Z_aex
np.random.seed(2645715)  # for reproducibility
Z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], (steps, N_simulations))

# Pre-allocate arrays to store the simulated paths
S_sx = np.zeros((steps + 1, N_simulations))
S_aex = np.zeros((steps + 1, N_simulations))
S_sx[0] = S_sx0
S_aex[0] = S_aex0

# Simulate the equity processes for SX5E and AEX
for t in range(1, steps + 1):
    S_sx[t] = S_sx[t-1] * np.exp((r - q - 0.5 * sigma_sx**2) * dt + sigma_sx * Z[t-1, :, 0] * np.sqrt(dt))
    S_aex[t] = S_aex[t-1] * np.exp((r - q - 0.5 * sigma_aex**2) * dt + sigma_aex * Z[t-1, :, 1] * np.sqrt(dt))

S_sx, S_aex


# Parameters for put options
K_SX5E, K_AEX = 3388, 616
# Correlation matrix
correlation_matrix = np.array([[1, rho], [rho, 1]])
# Cholesky decomposition
L = np.linalg.cholesky(correlation_matrix)

# Setting a seed for replicability
np.random.seed(2645715) # Student id

# Arrays to store the payoffs
put_SX5E = np.zeros(N_simulations)
put_AEX = np.zeros(N_simulations)

for i in range(N_simulations):
    # Generate correlated random variables for each simulation
    Z = np.random.randn(2)
    Z_correlated = L.dot(Z)
    
    # Simulated stock prices at maturity for SX5E and AEX
    S_SX5E_T_simulated = S_sx0 * exp((r - q - 0.5 * sigma_sx**2) * T + sigma_sx * Z_correlated[0] * sqrt(T))
    S_AEX_T_simulated = S_aex0 * exp((r - q - 0.5 * sigma_aex**2) * T + sigma_aex * Z_correlated[1] * sqrt(T))
    
    # Calculate the payoff for each option
    put_SX5E[i] = max(K_SX5E - S_SX5E_T_simulated, 0)
    put_AEX[i] = max(K_AEX - S_AEX_T_simulated, 0)

put_SX5E, put_AEX

###############################################################################
# a)
# Contract details
contracts = {
    'Forward_SX5E': {'type': 'forward', 'N': 10000, 'K': 4235},
    'Forward_AEX': {'type': 'forward', 'N': 55000, 'K': 770},
    'Put_SX5E': {'type': 'put', 'N': 10000, 'K': 3388},
    'Put_AEX': {'type': 'put', 'N': 55000, 'K': 616}
}

# Function to calculate contract value
def calculate_contract_value(S, contract, t):
    if contract['type'] == 'forward':
        # For forward, the value at maturity is N * (S - K)
        value = contract['N'] * (S[t] - contract['K'])
    elif contract['type'] == 'put':
        # For put, the value at maturity is N * max(K - S, 0)
        value = contract['N'] * np.maximum(contract['K'] - S[t], 0)
    return value

# Calculate contract values at maturity
portfolio_values = np.zeros((steps + 1, N_simulations))
for name, contract in contracts.items():
    if 'SX5E' in name:
        portfolio_values += calculate_contract_value(S_sx, contract, np.arange(steps + 1))
    else: # AEX contracts
        portfolio_values += calculate_contract_value(S_aex, contract, np.arange(steps + 1))

# Expected Positive Exposure (EPE) at each time step
EPE = np.maximum(portfolio_values, 0).mean(axis=1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(np.arange(steps + 1) / 12, EPE, label='Expected Positive Exposure')
plt.title('Monthly Expected Positive Exposure Profile for the Netted Portfolio')
plt.xlabel('Time (Years)')
plt.ylabel('Expected Positive Exposure')
plt.legend()
plt.grid(True)
plt.show()

'''
The plot above shows the monthly expected positive exposure profile for the 
netted portfolio of contracts over a 5-year period. 
This profile provides insight into the potential exposure the portfolio could face 
due to fluctuations in the underlying assets' prices. As expected, the exposure varies over time, 
reflecting the dynamics of the simulated equity processes and the nature of the financial instruments involved.
'''

###############################################################################
# Construct Excel example
###############################################################################

# Given time points T and corresponding lambda values
T = [0, 1, 2, 3, 4, 5]
lambda_values = [0.02, 0.02, 0.0215, 0.0215, 0.022, 0.022]

# Initialize cumulative intensity list with the first value as 0
cumulative_intensity = [0]

# Calculate the cumulative intensity for each year
for i in range(1, len(T)):
    cumulative_intensity.append(cumulative_intensity[-1] + lambda_values[i-1] * (T[i] - T[i-1]))

cumulative_intensity
'''
checked by hand: correct
'''

# Initialize the list for Q values
Q_values = []

# Calculate Q(T_{i-1}, T_i) for each interval
for i in range(1, len(T)):
    previous_cumulative_intensity = cumulative_intensity[i-1]
    current_cumulative_intensity = cumulative_intensity[i]
    Q = exp(-previous_cumulative_intensity) - exp(-current_cumulative_intensity)
    Q_values.append(Q)

Q_values
'''
checked by hand: correct
'''

###############################################################################
# b)

# Given lambda values and time points T
T = np.array([0, 1, 2, 3, 4, 5])
lambda_values = np.array([0.02, 0.02, 0.0215, 0.0215, 0.022, 0.022])

# Calculate cumulative intensity
cumulative_intensity = np.cumsum(np.diff(T, prepend=0) * lambda_values)

# Calculate Q values for each interval
Q_values = np.exp(-cumulative_intensity[:-1]) - np.exp(-cumulative_intensity[1:])

# Given LGD and risk-free rate r
LGD = 0.4
r = 0.03

# Calculate the discount factors for each timestep (monthly)
discount_factors = np.exp(-r * np.arange(S_sx.shape[0]) / 12)

# Calculate the positive exposure for SX5E forwards and AEX forwards, then the simulated losses
K_sx = contracts['Forward_SX5E']['K']
K_aex = contracts['Forward_AEX']['K']

positive_exposure_SX5E = np.maximum(S_sx - K_sx, 0)
simulated_losses_SX5E = positive_exposure_SX5E * LGD * discount_factors[:, np.newaxis]

positive_exposure_AEX = np.maximum(S_aex - K_aex, 0)
simulated_losses_AEX = positive_exposure_AEX * LGD * discount_factors[:, np.newaxis]

# Calculate the positive exposure for SX5E puts and AEX puts, then the simulated losses
K_put_SX5E = contracts['Put_SX5E']['K']
K_put_AEX = contracts['Put_AEX']['K']

positive_exposure_put_SX5E = np.maximum(K_put_SX5E - S_sx, 0)
simulated_losses_put_SX5E = positive_exposure_put_SX5E * LGD * discount_factors[:, np.newaxis]

positive_exposure_put_AEX = np.maximum(K_put_AEX - S_aex, 0)
simulated_losses_put_AEX = positive_exposure_put_AEX * LGD * discount_factors[:, np.newaxis]

#########################################
# Number of contracts for each instrument
N_Forward_SX5E = contracts['Forward_SX5E']['N']
N_Forward_AEX = contracts['Forward_AEX']['N']
N_Put_SX5E = contracts['Put_SX5E']['N']
N_Put_AEX = contracts['Put_AEX']['N']

# Adjust the simulated losses by the number of contracts
simulated_losses_SX5E *= N_Forward_SX5E
simulated_losses_AEX *= N_Forward_AEX
simulated_losses_put_SX5E *= N_Put_SX5E
simulated_losses_put_AEX *= N_Put_AEX
########################################

# Take the average discounted losses (conditional on default) for each timestep for all contracts
average_discounted_loss_SX5E = simulated_losses_SX5E.mean(axis=1)
average_discounted_loss_AEX = simulated_losses_AEX.mean(axis=1)
average_discounted_loss_put_SX5E = simulated_losses_put_SX5E.mean(axis=1)
average_discounted_loss_put_AEX = simulated_losses_put_AEX.mean(axis=1)

# To aggregate monthly losses into annual, we'll sum up the monthly values within each year
annual_discounted_loss_SX5E = np.array([average_discounted_loss_SX5E[T[i]*12:T[i+1]*12].sum() for i in range(len(T)-1)])
annual_discounted_loss_AEX = np.array([average_discounted_loss_AEX[T[i]*12:T[i+1]*12].sum() for i in range(len(T)-1)])
annual_discounted_loss_put_SX5E = np.array([average_discounted_loss_put_SX5E[T[i]*12:T[i+1]*12].sum() for i in range(len(T)-1)])
annual_discounted_loss_put_AEX = np.array([average_discounted_loss_put_AEX[T[i]*12:T[i+1]*12].sum() for i in range(len(T)-1)])

# Calculate CVA for each contract as the product of average discounted losses and Q_values
CVA_SX5E = annual_discounted_loss_SX5E * Q_values
CVA_AEX = annual_discounted_loss_AEX * Q_values
CVA_put_SX5E = annual_discounted_loss_put_SX5E * Q_values
CVA_put_AEX = annual_discounted_loss_put_AEX * Q_values

# Summing across time to get total CVA for each underlying and each contract type
CVA_SX5E_total = np.sum(CVA_SX5E)
CVA_AEX_total = np.sum(CVA_AEX)
CVA_put_SX5E_total = np.sum(CVA_put_SX5E)
CVA_put_AEX_total = np.sum(CVA_put_AEX)

CVA_Total_indipendent = CVA_SX5E_total+CVA_AEX_total+CVA_put_SX5E_total+CVA_put_AEX_total
print(CVA_SX5E_total,CVA_AEX_total, CVA_put_SX5E_total, CVA_put_AEX_total)
print(CVA_Total_indipendent)
###############################################################################
# c)
# Sum the CVA of all contracts to get the total CVA for the portfolio without netting
total_CVA_without_netting = CVA_SX5E_total + CVA_AEX_total + CVA_put_SX5E_total + CVA_put_AEX_total
print(total_CVA_without_netting)
###############################################################################
# d)

# net_positive_exposure is calculated considering the offsetting effects across all contracts
net_positive_exposure = (
    np.maximum(S_sx - K_sx, 0) * N_Forward_SX5E - np.maximum(K_put_SX5E - S_sx, 0) * N_Put_SX5E +
    np.maximum(S_aex - K_aex, 0) * N_Forward_AEX - np.maximum(K_put_AEX - S_aex, 0) * N_Put_AEX
)

# Apply LGD and discount factors
simulated_net_losses = net_positive_exposure * LGD * discount_factors[:, np.newaxis]

# Calculate expected net losses
average_net_loss = simulated_net_losses.mean(axis=1)

# Aggregate losses over the intervals defined by the Q values
annual_net_loss = np.array([average_net_loss[T[i]*12:T[i+1]*12].sum() for i in range(len(T)-1)])

# Calculate CVA using Q values
CVA_netted = annual_net_loss * Q_values

# Total CVA for the portfolio under netting agreement
total_CVA_netted_portfolio = CVA_netted.sum()
print(total_CVA_netted_portfolio)















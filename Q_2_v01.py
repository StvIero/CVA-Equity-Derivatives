# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:16:50 2024

@author: ieron
"""

import matplotlib.pyplot as plt
import numpy as np
from math import exp, sqrt

# Simplified simulation parameters
T = 5  # years
dt = 1/12  # monthly steps
n_steps = int(T / dt)
r = 0.03  # risk-free rate
q = 0.02  # dividend yield
sigma = 0.15  # assumed volatility for both SX5E and AEX
rho = 0.8  # correlation

# Correlation matrix and Cholesky decomposition
correlation_matrix = np.array([[1, rho], [rho, 1]])
L = np.linalg.cholesky(correlation_matrix)

# Initial prices
S_SX5E_0 = 4235
S_AEX_0 = 770

# Set simulations
num_simulations=10000

# Simulate price paths
np.random.seed(2645715)  # For reproducibility
price_paths_SX5E = np.zeros((n_steps + 1, num_simulations))
price_paths_AEX = np.zeros((n_steps + 1, num_simulations))
price_paths_SX5E[0, :] = S_SX5E_0
price_paths_AEX[0, :] = S_AEX_0

for t in range(1, n_steps + 1):
    Z = np.random.randn(2, num_simulations)
    Z_correlated = L.dot(Z)
    
    price_paths_SX5E[t, :] = price_paths_SX5E[t-1, :] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * Z_correlated[0])
    price_paths_AEX[t, :] = price_paths_AEX[t-1, :] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * Z_correlated[1])

# For simplicity, let's assume the netted portfolio value equals the sum of the two price paths
# This is a simplification and not reflective of actual contract valuations
netted_portfolio_value = price_paths_SX5E + price_paths_AEX

# Calculate expected exposure (EE) as the positive part of the netted portfolio value, averaged across simulations
expected_exposure = np.maximum(0, netted_portfolio_value).mean(axis=1)

# Plot the expected exposure profile
months = np.arange(0, n_steps + 1) * dt
plt.figure(figsize=(10, 6))
plt.plot(months, expected_exposure, label='Expected Exposure', color='blue')
plt.title('Monthly Expected Exposure Profile for Netted Portfolio')
plt.xlabel('Time (Years)')
plt.ylabel('Expected Exposure')
plt.legend()
plt.grid(True)
plt.show()

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
###############################################################################
# Re-establishing the parameters for the calculation
LGD = 0.4  # Loss Given Default
r = 0.03  # Risk-free rate
q = 0.02
T = [0, 1, 2, 3, 4, 5]  # Maturity in years
S_SX5E_0 = 4235  # Initial price for SX5E
S_AEX_0 = 770  # Initial price for AEX
num_simulations = 100000  # Number of simulations
sigma_SX5E = 0.15  # Volatility for SX5E
sigma_AEX = 0.15  # Volatility for AEX
K_SX5E = 4235
K_AEX = 770

# Set seed for reproducibility
np.random.seed(2645715)

# Initialize arrays to store simulated prices for each time step
simulated_prices_SX5E = np.zeros((num_simulations, len(T)))
simulated_prices_AEX = np.zeros((num_simulations, len(T)))

# Set initial prices for the first time step
simulated_prices_SX5E[:, 0] = S_SX5E_0
simulated_prices_AEX[:, 0] = S_AEX_0

# Simulate the stock prices at each time step for both SX5E and AEX
for t in range(1, len(T)):
    dt = 1  # Assuming 1 year time steps
    Z_SX5E = np.random.randn(num_simulations)
    Z_AEX = np.random.randn(num_simulations)

    simulated_prices_SX5E[:, t] = simulated_prices_SX5E[:, t-1] * np.exp(
        (r - q - 0.5 * sigma_SX5E**2) * dt + sigma_SX5E * Z_SX5E * sqrt(dt))
    simulated_prices_AEX[:, t] = simulated_prices_AEX[:, t-1] * np.exp(
        (r - q - 0.5 * sigma_AEX**2) * dt + sigma_AEX * Z_AEX * sqrt(dt))

# Calculate positive exposures and discount them for each time step
discount_factors = np.array([exp(-r * t) for t in T])
positive_exposure_SX5E = np.maximum(simulated_prices_SX5E - K_SX5E, 0)
discounted_positive_exposure_SX5E = positive_exposure_SX5E * discount_factors
simulated_losses_SX5E = discounted_positive_exposure_SX5E * LGD

positive_exposure_AEX = np.maximum(simulated_prices_AEX - K_AEX, 0)
discounted_positive_exposure_AEX = positive_exposure_AEX * discount_factors
simulated_losses_AEX = discounted_positive_exposure_AEX * LGD

# Calculate the average simulated losses for each time step
average_simulated_loss_SX5E = np.mean(simulated_losses_SX5E, axis=0)
average_simulated_loss_SX5E

average_simulated_loss_AEX = np.mean(simulated_losses_AEX, axis=0)
average_simulated_loss_AEX


# Calculate CVA for each T without summing across all time steps
CVA_SX5E = average_simulated_loss_SX5E[1:] * Q_values
print("CVA for the Forward SX5E:", CVA_SX5E)

CVA_AEX = average_simulated_loss_AEX[1:] * Q_values
print("CVA for the Forward AEX:", CVA_AEX)

###############################################################################
###############################################################################
# Adjust strike prices for put options
K_SX5E_put = 3388
K_AEX_put = 616

# Set seed for reproducibility
np.random.seed(2645715)

# Initialize arrays to hold simulated put option losses and CVA calculations
simulated_losses_put_SX5E = np.zeros((num_simulations, len(T)))
simulated_losses_put_AEX = np.zeros((num_simulations, len(T)))
CVA_put_SX5E = np.zeros(len(T)-1)
CVA_put_AEX = np.zeros(len(T)-1)

for i in range(1, len(T)):
    # Positive put option exposure
    positive_exposure_put_SX5E = np.maximum(K_SX5E_put - simulated_prices_SX5E[:, i], 0)
    positive_exposure_put_AEX = np.maximum(K_AEX_put - simulated_prices_AEX[:, i], 0)
    
    # Discounted positive exposure and simulated losses
    discounted_positive_exposure_put_SX5E = positive_exposure_put_SX5E * discount_factors[i]
    discounted_positive_exposure_put_AEX = positive_exposure_put_AEX * discount_factors[i]
    simulated_losses_put_SX5E[:, i] = discounted_positive_exposure_put_SX5E * LGD
    simulated_losses_put_AEX[:, i] = discounted_positive_exposure_put_AEX * LGD
    
    # Average simulated losses for CVA calculation
    average_simulated_loss_put_SX5E = np.mean(simulated_losses_put_SX5E[:, i])
    average_simulated_loss_put_AEX = np.mean(simulated_losses_put_AEX[:, i])
    
    # CVA for each time step
    CVA_put_SX5E[i-1] = average_simulated_loss_put_SX5E * Q_values[i-1]
    CVA_put_AEX[i-1] = average_simulated_loss_put_AEX * Q_values[i-1]

# Displaying the CVA for put options at each time step
print("CVA for SX5E Put Option at each T:", CVA_put_SX5E)
print("CVA for AEX Put Option at each T:", CVA_put_AEX)

###############################################################################
###############################################################################
K_SX5E = 4235
K_AEX = 770
K_SX5E_put = 3388
K_AEX_put = 616

# Initialize arrays for CVA calculations
CVA_Forward_SX5E, CVA_Forward_AEX, CVA_Put_SX5E, CVA_Put_AEX = (np.zeros(len(T)-1) for _ in range(4))

# Set seed for reproducibility
np.random.seed(2645715)

for i in range(1, len(T)):
    # Calculate positive exposures and discount them
    # Forwards
    positive_exp_Forward_SX5E = np.maximum(simulated_prices_SX5E[:, i] - S_SX5E_0, 0)
    positive_exp_Forward_AEX = np.maximum(simulated_prices_AEX[:, i] - S_AEX_0, 0)
    # Puts
    positive_exp_Put_SX5E = np.maximum(K_SX5E_put - simulated_prices_SX5E[:, i], 0)
    positive_exp_Put_AEX = np.maximum(K_AEX_put - simulated_prices_AEX[:, i], 0)
    
    # Discount positive exposures
    discounted_exp_Forward_SX5E = positive_exp_Forward_SX5E * discount_factors[i]
    discounted_exp_Forward_AEX = positive_exp_Forward_AEX * discount_factors[i]
    discounted_exp_Put_SX5E = positive_exp_Put_SX5E * discount_factors[i]
    discounted_exp_Put_AEX = positive_exp_Put_AEX * discount_factors[i]
    
    # Simulated losses
    losses_Forward_SX5E = discounted_exp_Forward_SX5E * LGD
    losses_Forward_AEX = discounted_exp_Forward_AEX * LGD
    losses_Put_SX5E = discounted_exp_Put_SX5E * LGD
    losses_Put_AEX = discounted_exp_Put_AEX * LGD
    
    # CVA calculation for each instrument
    CVA_Forward_SX5E[i-1] = np.mean(losses_Forward_SX5E) * Q_values[i-1]
    CVA_Forward_AEX[i-1] = np.mean(losses_Forward_AEX) * Q_values[i-1]
    CVA_Put_SX5E[i-1] = np.mean(losses_Put_SX5E) * Q_values[i-1]
    CVA_Put_AEX[i-1] = np.mean(losses_Put_AEX) * Q_values[i-1]

# Aggregate CVA Charges for the Portfolio
Portfolio_CVA = CVA_Forward_SX5E + CVA_Forward_AEX + CVA_Put_SX5E + CVA_Put_AEX

# Print Portfolio CVA for each T
print("Portfolio CVA at each T:", Portfolio_CVA)

# Check manually
# OK 

###############################################################################

# Initialize arrays for net exposures and CVA calculation under netting agreement
net_exposures = np.zeros((num_simulations, len(T)-1))
CVA_netted_portfolio = np.zeros(len(T)-1)

for i in range(1, len(T)):
    # Aggregate exposures for all instruments at time step i
    net_exposure_i = (
        np.maximum(simulated_prices_SX5E[:, i] - S_SX5E_0, 0) +  # Forward_SX5E exposure
        np.maximum(simulated_prices_AEX[:, i] - S_AEX_0, 0) -    # Forward_AEX exposure
        np.maximum(K_SX5E_put - simulated_prices_SX5E[:, i], 0) -  # Put_SX5E exposure (negative for puts)
        np.maximum(K_AEX_put - simulated_prices_AEX[:, i], 0)    # Put_AEX exposure (negative for puts)
    )
    
    # Ensure only positive net exposures contribute to CVA
    positive_net_exposure_i = np.maximum(net_exposure_i, 0)
    
    # Discount net exposure at time step i
    discounted_net_exposure_i = positive_net_exposure_i * discount_factors[i]
    
    # Calculate net simulated losses
    net_simulated_losses_i = discounted_net_exposure_i * LGD
    
    # Average net simulated losses for CVA calculation
    average_net_simulated_loss_i = np.mean(net_simulated_losses_i)
    
    # CVA for the netted portfolio at time step i
    CVA_netted_portfolio[i-1] = average_net_simulated_loss_i * Q_values[i-1]

# Display CVA for the netted portfolio at each T
print("CVA for the netted portfolio at each T:", CVA_netted_portfolio)

###############################################################################
###############################################################################
###############################################################################

















# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:40:30 2024

@author: ieron
"""
import numpy as np
from math import exp, sqrt

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

###############################################################################
###############################################################################
###############################################################################

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

frequencies_months = [1, 2, 3, 6, 12, 24, 36, 48, 60]  # Collateral posting frequencies in months

# Placeholder for CVA charges for each frequency
cva_charges = []

# Function to calculate CVA (Placeholder, replace with your actual CVA calculation logic)
def calculate_cva_with_collateral(frequency):
    # Placeholder for adjusted CVA calculation
    # Implement the logic to adjust for collateral based on the frequency
    return np.random.rand()  # Return a random number as a placeholder

# Calculate CVA for each collateral posting frequency
for freq in frequencies_months:
    cva = calculate_cva_with_collateral(freq)
    cva_charges.append(cva)

# Calculate the "no-collateral" CVA charge as a reference
no_collateral_cva = calculate_cva_with_collateral(None)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(frequencies_months, cva_charges, marker='o', linestyle='-', color='b', label='With Collateral')
plt.axhline(y=no_collateral_cva, color='r', linestyle='--', label='No Collateral')
plt.title('Impact of Collateral Posting Frequency on CVA Charge')
plt.xlabel('Collateral Posting Frequency (months)')
plt.ylabel('CVA Charge')
plt.legend()
plt.xticks(frequencies_months + [0])  # Adding 0 for "no-collateral" reference
plt.grid(True)
plt.show()



























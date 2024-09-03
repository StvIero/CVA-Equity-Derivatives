# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:50:27 2024

@author: ieron
"""
import numpy as np
from math import exp, sqrt

# Adjusted volatility
sigma = 0.3  # Increasing volatility to 30%
LGD = 0.4  # Loss Given Default
r = 0.03  # Risk-free rate
q = 0.02  # Divident yield
T = np.array([0, 1, 2, 3, 4, 5])  # Maturity in years
S_SX5E_0 = 4235  # Initial price for SX5E
S_AEX_0 = 770  # Initial price for AEX
num_simulations = 100000  # Number of simulations
K_SX5E_put = 3388
K_AEX_put = 616

# Set seed for reproducibility
np.random.seed(2645715)

# Initialize arrays to store simulated prices for each time step
simulated_prices_SX5E = np.zeros((num_simulations, len(T)))
simulated_prices_AEX = np.zeros((num_simulations, len(T)))

# Set initial prices for the first time step
simulated_prices_SX5E[:, 0] = S_SX5E_0
simulated_prices_AEX[:, 0] = S_AEX_0

# Simulate the stock prices at each time step for both SX5E and AEX with updated volatility
for t in range(1, len(T)):
    dt = T[t] - T[t-1]  # Time step
    Z_SX5E = np.random.randn(num_simulations)
    Z_AEX = np.random.randn(num_simulations)

    simulated_prices_SX5E[:, t] = simulated_prices_SX5E[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * Z_SX5E * sqrt(dt))
    simulated_prices_AEX[:, t] = simulated_prices_AEX[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * Z_AEX * sqrt(dt))

'''
 discount_factors, Q_values are defined from previous question
'''

# Initialize arrays for net exposures and CVA calculation under netting agreement
net_exposures = np.zeros((num_simulations, len(T)-1))
CVA_netted_portfolio = np.zeros(len(T)-1)

for i in range(1, len(T)):
    # Aggregate exposures for all instruments at time step i
    net_exposure_i = (
        (np.maximum(simulated_prices_SX5E[:, i] - S_SX5E_0, 0)*10000) +  # Forward_SX5E exposure
        (np.maximum(simulated_prices_AEX[:, i] - S_AEX_0, 0)*55000) -    # Forward_AEX exposure
        (np.maximum(K_SX5E_put - simulated_prices_SX5E[:, i], 0)*10000) -  # Put_SX5E exposure (negative for puts)
        (np.maximum(K_AEX_put - simulated_prices_AEX[:, i], 0) *55000)   # Put_AEX exposure (negative for puts)
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
print("CVA for the netted portfolio at each T with increased volatility:", CVA_netted_portfolio)
# [4.56621293 6.14369496 7.73752696 8.54318616 9.39246575]

###############################################################################

# Re-importing necessary libraries and re-defining variables due to reset
import numpy as np
from math import exp, sqrt

# Adjusted parameters
sigma = 0.15  # Volatility
rho = 0.4  # Adjusted correlation for log-returns
LGD = 0.4  # Loss Given Default
r = 0.03  # Risk-free rate
q = 0.02  # Dividend yield
T = np.array([0, 1, 2, 3, 4, 5])  # Maturity in years
S_SX5E_0 = 4235  # Initial price for SX5E
S_AEX_0 = 770  # Initial price for AEX
num_simulations = 100000  # Number of simulations
K_SX5E_put = 3388
K_AEX_put = 616
discount_factors = np.exp(-r * T[1:])  # Discount factors for each time step excluding T=0
Q_values = np.array([0.01980133, 0.01940923, 0.02043649, 0.02000180, 0.02002662])  # From previous question

# Set seed for reproducibility
np.random.seed(2645715)

# Cholesky decomposition for the correlation matrix
correlation_matrix = np.array([[1, rho], [rho, 1]])
L = np.linalg.cholesky(correlation_matrix)

# Initialize arrays to store simulated prices for each time step
simulated_prices_SX5E = np.zeros((num_simulations, len(T)))
simulated_prices_AEX = np.zeros((num_simulations, len(T)))
simulated_prices_SX5E[:, 0] = S_SX5E_0  # Set initial values
simulated_prices_AEX[:, 0] = S_AEX_0  # Set initial values

# Simulate the stock prices with adjusted correlation for log-returns
for t in range(1, len(T)):
    dt = T[t] - T[t-1]  # Time step
    Z = np.random.randn(num_simulations, 2)  # Generate random variables for both SX5E and AEX
    Z_correlated = Z.dot(L.T)  # Apply Cholesky decomposition to get correlated random variables

    simulated_prices_SX5E[:, t] = simulated_prices_SX5E[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * Z_correlated[:, 0] * sqrt(dt))
    simulated_prices_AEX[:, t] = simulated_prices_AEX[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * Z_correlated[:, 1] * sqrt(dt))

# Calculate CVA for netted portfolio with adjusted correlation
CVA_netted_portfolio = np.zeros(len(T)-1)

for i in range(1, len(T)):
    net_exposure_i = (
        (np.maximum(simulated_prices_SX5E[:, i] - S_SX5E_0, 0)*10000) +  
        (np.maximum(simulated_prices_AEX[:, i] - S_AEX_0, 0)*55000) -    
        (np.maximum(K_SX5E_put - simulated_prices_SX5E[:, i], 0)*10000) - 
        (np.maximum(K_AEX_put - simulated_prices_AEX[:, i], 0) *55000)   
    )
    
    positive_net_exposure_i = np.maximum(net_exposure_i, 0)
    discounted_net_exposure_i = positive_net_exposure_i * discount_factors[i-1]
    net_simulated_losses_i = discounted_net_exposure_i * LGD
    average_net_simulated_loss_i = np.mean(net_simulated_losses_i)
    CVA_netted_portfolio[i-1] = average_net_simulated_loss_i * Q_values[i-1]

print("CVA for the netted portfolio at each T with reduced correlation:",CVA_netted_portfolio)
# [2.50452063, 3.47940554, 4.48542462, 5.03419538, 5.59965949]

###############################################################################

'''
Increased Volatility Results:
The CVA values are higher across all future time steps compared to the baseline scenario. 
This increase is due to the higher volatility (σ=0.3σ=0.3 or 30%) used in the simulations. 
Higher volatility leads to a greater range of potential outcomes for the underlying asset prices, 
increasing the potential for both higher gains and losses. In the context of CVA, 
which focuses on positive net exposures that could be lost in the event of counterparty default, 
higher volatility increases the potential positive net exposure, thus increasing the CVA.

Decreased Correlation Results:
Lower CVA values are observed across all time steps when the correlation between 
the underlying assets is decreased from 0.8 to 0.4. 
Decreasing the correlation means the price movements of the underlying assets become less synchronized. 
In a diversified portfolio, this can lead to a natural hedging effect 
where losses in one asset may be offset by gains in another, 
reducing the net exposure of the portfolio to positive outcomes that are at risk in the event of a counterparty default. 
Consequently, the CVA, which quantifies the risk of these positive net exposures, is lower.

Impact of Volatility: 
The results reinforce the principle that higher volatility increases the credit risk quantified by CVA, 
as it amplifies the range of potential positive exposures at risk in case of default.

Impact of Correlation: 
The decrease in correlation demonstrates how less synchronized movements 
between assets can reduce credit risk in a netted portfolio, 
showcasing the importance of diversification and the hedging potential within a portfolio.

The outcomes illustrate the critical roles both volatility and correlation 
play in determining the credit risk of a portfolio under a netting agreement. 
While higher volatility increases CVA by widening the potential distribution of exposures, 
lower correlation can mitigate credit risk by reducing the portfolio's net positive exposure, 
leading to lower CVA charges. These insights emphasize the need for careful risk management 
and the benefits of diversification within financial portfolios.
'''










































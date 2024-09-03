# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:34:17 2024

@author: ieron
"""
import numpy as np
from math import exp, sqrt

# Re-defining given values for clarity
r = 0.03  # risk-free rate
q = 0.02  # dividend yield
sigma_SX5E = 0.15  # volatility for SX5E
sigma_AEX = 0.15  # volatility for AEX
T = 5  # maturity in years
S_SX5E_0 = 4235  # initial price for SX5E
S_AEX_0 = 770  # initial price for AEX
num_simulations = 100000  # number of Monte Carlo simulations
rho = 0.8  # Given correlation

# Correlation matrix
correlation_matrix = np.array([[1, rho], [rho, 1]])

# Cholesky decomposition
L = np.linalg.cholesky(correlation_matrix)

# Setting a seed for replicability
np.random.seed(2645715) # Student id

# Adjusted Monte Carlo simulation to include correlation
S_SX5E_T_correlated_simulations = np.zeros(num_simulations)
S_AEX_T_correlated_simulations = np.zeros(num_simulations)

for i in range(num_simulations):
    # Generate independent standard normal random variables
    Z = np.random.randn(2)
    # Apply the Cholesky decomposition to obtain correlated random variables
    Z_correlated = L.dot(Z)
    
    # Use the correlated random variables for SX5E and AEX
    S_SX5E_T_simulated = S_SX5E_0 * exp((r - q - 0.5 * sigma_SX5E**2) * T + sigma_SX5E * Z_correlated[0] * sqrt(T))
    S_AEX_T_simulated = S_AEX_0 * exp((r - q - 0.5 * sigma_AEX**2) * T + sigma_AEX * Z_correlated[1] * sqrt(T))

    S_SX5E_T_correlated_simulations[i] = S_SX5E_T_simulated
    S_AEX_T_correlated_simulations[i] = S_AEX_T_simulated

# Calculate average expected price at maturity for correlated simulations
average_S_SX5E_T_correlated = np.mean(S_SX5E_T_correlated_simulations)
average_S_AEX_T_correlated = np.mean(S_AEX_T_correlated_simulations)

# Discounting the average expected prices at maturity to present value
PV_S_SX5E_T_correlated = average_S_SX5E_T_correlated * exp((-r+q) * T)
PV_S_AEX_T_correlated = average_S_AEX_T_correlated * exp((-r +q) * T)

PV_S_SX5E_T_correlated, PV_S_AEX_T_correlated

'''
 (4236.3246742380425, 770.2846485123276)
 vs (4235 , 770)  gives 
 diff = [1.324674238042462 , 0.2846485123276352]  'I am happy with that :)'
'''

# Calculate standard deviations of the simulated outcomes
std_dev_S_SX5E = np.std(S_SX5E_T_correlated_simulations)
std_dev_S_AEX = np.std(S_AEX_T_correlated_simulations)

# Calculate SEM for both simulations
SEM_S_SX5E = std_dev_S_SX5E / sqrt(num_simulations)
SEM_S_AEX = std_dev_S_AEX / sqrt(num_simulations)

# Z-score for 95% confidence
Z = 1.96

# Calculate 95% confidence intervals
CI_S_SX5E = (PV_S_SX5E_T_correlated - Z * SEM_S_SX5E, PV_S_SX5E_T_correlated + Z * SEM_S_SX5E)
CI_S_AEX = (PV_S_AEX_T_correlated - Z * SEM_S_AEX, PV_S_AEX_T_correlated + Z * SEM_S_AEX)

CI_S_SX5E, CI_S_AEX
'''
(4226.8205084621095, 4245.828840013975),
 (768.5509965782467, 772.0183004464086)
'''

###############################################################################
###############################################################################
from scipy.stats import norm

# Function to calculate Black-Scholes price for put options
def black_scholes_put(S0, K, T, r, q, sigma):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * exp(-r * T) * norm.cdf(-d2) - S0 * exp(-q * T) * norm.cdf(-d1)
    return put_price

# Parameters for SX5E and AEX put options
S0_SX5E, K_SX5E, T, r, q , sigma_SX5E = 4235, 3388, 5, 0.03, 0.02 , 0.15
S0_AEX, K_AEX, T, r, q , sigma_AEX = 770, 616 , 5, 0.03, 0.02 , 0.15

# Calculate Black-Scholes prices
BS_price_SX5E_put = black_scholes_put(S0_SX5E, K_SX5E, T, r, q, sigma_SX5E)
BS_price_AEX_put = black_scholes_put(S0_AEX, K_AEX, T, r, q, sigma_AEX)

BS_price_SX5E_put, BS_price_AEX_put
# (130.3097531394726, 23.69268238899504)
###############################################################################
###############################################################################

# Parameters for put options
K_SX5E, K_AEX = 3388, 616

# Setting a seed for replicability
np.random.seed(2645715) # Student id

# Arrays to store the payoffs
payoffs_SX5E = np.zeros(num_simulations)
payoffs_AEX = np.zeros(num_simulations)

for i in range(num_simulations):
    # Generate correlated random variables for each simulation
    Z = np.random.randn(2)
    Z_correlated = L.dot(Z)
    
    # Simulated stock prices at maturity for SX5E and AEX
    S_SX5E_T_simulated = S_SX5E_0 * exp((r - q - 0.5 * sigma_SX5E**2) * T + sigma_SX5E * Z_correlated[0] * sqrt(T))
    S_AEX_T_simulated = S_AEX_0 * exp((r - q - 0.5 * sigma_AEX**2) * T + sigma_AEX * Z_correlated[1] * sqrt(T))
    
    # Calculate the payoff for each option
    payoffs_SX5E[i] = max(K_SX5E - S_SX5E_T_simulated, 0)
    payoffs_AEX[i] = max(K_AEX - S_AEX_T_simulated, 0)

# Average the payoffs and discount them to present value
expected_payoff_SX5E = np.mean(payoffs_SX5E) * exp(-r  * T)
expected_payoff_AEX = np.mean(payoffs_AEX) * exp(-r  * T)

expected_payoff_SX5E, expected_payoff_AEX
'''
 (130.68169839089506, 23.78583514804382)
 vs (130.3097531394726, 23.69268238899504)
 diff = (0.37194525142245993, 0.09315275904878106) 'Happy with that aswell :)'
'''

# Z-score for 95% confidence
Z = 1.96

# Calculate standard deviations of the payoffs
std_dev_payoffs_SX5E = np.std(payoffs_SX5E)
std_dev_payoffs_AEX = np.std(payoffs_AEX)

# Calculate SEM for both sets of payoffs
SEM_payoffs_SX5E = std_dev_payoffs_SX5E / sqrt(num_simulations)
SEM_payoffs_AEX = std_dev_payoffs_AEX / sqrt(num_simulations)

# Calculate 95% confidence intervals
CI_payoffs_SX5E = (expected_payoff_SX5E - Z * SEM_payoffs_SX5E, expected_payoff_SX5E + Z * SEM_payoffs_SX5E)
CI_payoffs_AEX = (expected_payoff_AEX - Z * SEM_payoffs_AEX, expected_payoff_AEX + Z * SEM_payoffs_AEX)

CI_payoffs_SX5E, CI_payoffs_AEX
'''
(128.59613950214654, 132.7672572796436),
 (23.406045313001222, 24.165624983086417)
'''

###############################################################################

# Calculate log-returns for SX5E and AEX
log_returns_SX5E = np.log(S_SX5E_T_correlated_simulations / S_SX5E_0)
log_returns_AEX = np.log(S_AEX_T_correlated_simulations / S_AEX_0)

# Calculate the correlation between the log-returns of SX5E and AEX
correlation_log_returns = np.corrcoef(log_returns_SX5E, log_returns_AEX)[0, 1]

correlation_log_returns
'''
 0.799907533689217 
 vs 0.8
 diff = -0.000092466 'Happy happy happy :)'
'''

# Fisher transformation of the correlation coefficient
z = 0.5 * np.log((1 + correlation_log_returns) / (1 - correlation_log_returns))

# Standard error for the Fisher transformation
SE_z = 1 / np.sqrt(num_simulations - 3)

# 95% confidence interval for the Fisher-transformed value
z_CI_lower = z - Z * SE_z
z_CI_upper = z + Z * SE_z

# Inverse Fisher transformation for the confidence interval
r_CI_lower = (np.exp(2 * z_CI_lower) - 1) / (np.exp(2 * z_CI_lower) + 1)
r_CI_upper = (np.exp(2 * z_CI_upper) - 1) / (np.exp(2 * z_CI_upper) + 1)

r_CI_lower, r_CI_upper

'''
(0.7974156193292502, 0.8112519329197673)
'''


































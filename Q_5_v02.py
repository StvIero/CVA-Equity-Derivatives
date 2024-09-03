# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:51:54 2024

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
Q_values = np.array([0.01980133, 0.01940923, 0.02043649, 0.02000180, 0.02002662])  # From previous question
discount_factors = np.array([exp(-r * t) for t in T])

net_exposures = np.zeros((num_simulations, len(T)-1))
CVA_netted_portfolio = np.zeros(len(T)-1)

for i in range(1, len(T)):
    # Aggregate exposures for all instruments at time step i
    net_exposure_i = (
        (np.maximum(simulated_prices_SX5E[:, i] - S_SX5E_0, 0)*10000) +  # Forward_SX5E exposure
        (np.maximum(simulated_prices_AEX[:, i] - S_AEX_0, 0)*55000) -    # Forward_AEX exposure
        (np.maximum(K_SX5E_put - simulated_prices_SX5E[:, i], 0)*10000) -  # Put_SX5E exposure (negative for puts)
        (np.maximum(K_AEX_put - simulated_prices_AEX[:, i], 0)*55000)    # Put_AEX exposure (negative for puts)
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

# Adjust the hazard rates by 10 bps in the specified intervals and recalculate the CVA
hazard_rate_adjustments = [0.001, 0.001, 0.001]  # 10 bps increase for each interval
# Initialize the list to store the change in CVA for each scenario
adjusted_cva_changes = []

# Loop over each hazard rate adjustment scenario
for scenario, adjustment in enumerate(hazard_rate_adjustments):
    # Adjust the Q_values based on the scenario
    adjusted_Q_values = np.array(Q_values)
    if scenario == 0:  # Adjusting for the [0,1] year interval
        adjusted_Q_values[0] += adjustment
    elif scenario == 1:  # Adjusting for the [1,3] year interval
        adjusted_Q_values[1:3] += adjustment
    else:  # Adjusting for the [3,5] year interval
        adjusted_Q_values[3:5] += adjustment

    # Recalculate CVA for the adjusted hazard rates
    adjusted_cva_netted_portfolio = np.zeros(len(T)-1)
    for i in range(1, len(T)):
        # Calculate the net exposures with the adjusted hazard rates
        adjusted_net_exposures = (
            np.maximum(simulated_prices_SX5E[:, i] - S_SX5E_0, 0)*10000 +  
            np.maximum(simulated_prices_AEX[:, i] - S_AEX_0, 0)*55000 -    
            np.maximum(K_SX5E_put - simulated_prices_SX5E[:, i], 0)*10000 -  
            np.maximum(K_AEX_put - simulated_prices_AEX[:, i], 0)*55000    
        )
        positive_adjusted_net_exposure_i = np.maximum(adjusted_net_exposures, 0)
        discounted_adjusted_net_exposure_i = positive_adjusted_net_exposure_i * discount_factors[i]
        adjusted_net_simulated_losses_i = discounted_adjusted_net_exposure_i * LGD
        adjusted_average_net_simulated_loss_i = np.mean(adjusted_net_simulated_losses_i)
        adjusted_cva_netted_portfolio[i-1] = adjusted_average_net_simulated_loss_i * adjusted_Q_values[i-1]

    # Calculate the change in CVA charge for this scenario
    cva_change = adjusted_cva_netted_portfolio - CVA_netted_portfolio
    adjusted_cva_changes.append(np.sum(cva_change))  # Summing the changes for simplicity

# Output the change in CVA charge for each hazard rate increase scenario
adjusted_cva_changes
# [2102.064266053436, 6428.020005798033, 8408.94760371407]


###############################################################################
###############################################################################
###############################################################################
import numpy as np
from scipy.integrate import quad

# Given parameters
R = 0.01  # Fixed rate of the CDS
LGD = 0.4  # Loss Given Default
r = 0.03  # Risk-free interest rate
T = 5  # Maturity of the CDS in years
N = 5  # Number of payment periods, assuming annual payments for simplicity

# Time points for annual premiums
T_i = np.linspace(0, T, N+1)[1:]  
T_i_with_start = np.insert(T_i, 0, 0)  # Include 0 for the start
Tmid_i = (T_i_with_start[:-1] + T_i_with_start[1:]) / 2  # Midpoints for calculations

# Function to calculate the survival probability Q(tau > T) using piecewise constant hazard rates
def survival_probability(T, average_lambda):
    return np.exp(-average_lambda * T)

# CDS pricing function based on the provided formula
def CDS_pricing(R, LGD, r, T_i, Tmid_i, average_lambda):
    # Calculate survival probabilities
    Q_tau_Ti = np.array([survival_probability(t, average_lambda) for t in T_i])
    Q_tau_Ti_minus_1 = np.array([survival_probability(t, average_lambda) for t in np.insert(T_i, 0, 0)[:-1]])
    
    # Calculate the sums for the premium and protection legs
    Sum_1 = np.sum(np.exp(-r * T_i) * (T_i - np.insert(T_i, 0, 0)[:-1]) * Q_tau_Ti)
    Sum_2 = np.sum(np.exp(-r * Tmid_i) * (Q_tau_Ti_minus_1 - Q_tau_Ti) * (T_i - np.insert(T_i, 0, 0)[:-1]) / 2)
    Sum_3 = np.sum(np.exp(-r * Tmid_i) * (Q_tau_Ti_minus_1 - Q_tau_Ti))
    
    # Calculate the CDS price
    PV_Premium_Leg = R * (Sum_1 + Sum_2)
    PV_Protection_Leg = LGD * Sum_3
    
    return PV_Premium_Leg - PV_Protection_Leg

# Initial average hazard rate (for simplicity, assuming a single rate for all intervals)
initial_average_lambda = 0.02

# Calculate the CDS price under the initial hazard rate
initial_cds_price = CDS_pricing(R, LGD, r, T_i, Tmid_i, initial_average_lambda)

# Calculate the change in CDS price for a 10 bps increase in the hazard rate
delta_lambda = 0.001  # 10 bps increase
adjusted_average_lambda = initial_average_lambda + delta_lambda
adjusted_cds_price = CDS_pricing(R, LGD, r, T_i, Tmid_i, adjusted_average_lambda)

# Calculate the change in CDS price
cds_price_change = adjusted_cds_price - initial_cds_price

print(f"Initial CDS Price: {initial_cds_price:.4f}")
print(f"Adjusted CDS Price (after 10 bps increase): {adjusted_cds_price:.4f}")
print(f"Change in CDS Price due to 10 bps increase in hazard rate: {cds_price_change:.4f}")

# Function to calculate CDS price for specific maturity and hazard rate adjustments
def calculate_cds_for_maturity(maturity, initial_lambda, adjustment_intervals):
    T_i_specific = np.linspace(0, maturity, maturity*N//T + 1)[1:]  # Adjust time points for maturity
    T_i_with_start_specific = np.insert(T_i_specific, 0, 0)
    Tmid_i_specific = (T_i_with_start_specific[:-1] + T_i_with_start_specific[1:]) / 2
    
    # Adjust the hazard rate based on the specified intervals
    adjusted_lambda = initial_lambda + sum([adjustment_intervals.get(i, 0) for i in range(1, maturity+1)])
    
    # Calculate and return the CDS price for the specific maturity
    return CDS_pricing(R, LGD, r, T_i_specific, Tmid_i_specific, adjusted_lambda)

# Initial hazard rate adjustments for each scenario
adjustment_intervals_0_1 = {1: delta_lambda}  # Increase in [0,1] affects all maturities
adjustment_intervals_1_3 = {2: delta_lambda, 3: delta_lambda}  # Increase in [1,3] affects 3Y and 5Y
adjustment_intervals_3_5 = {4: delta_lambda, 5: delta_lambda}  # Increase in [3,5] affects only 5Y

# Calculate CDS price changes for each maturity and scenario
cds_price_changes = {}
for maturity in [1, 3, 5]:
    initial_price = calculate_cds_for_maturity(maturity, initial_average_lambda, {})
    price_change_0_1 = calculate_cds_for_maturity(maturity, initial_average_lambda, adjustment_intervals_0_1) - initial_price
    price_change_1_3 = calculate_cds_for_maturity(maturity, initial_average_lambda, adjustment_intervals_1_3) - initial_price if maturity > 1 else 0
    price_change_3_5 = calculate_cds_for_maturity(maturity, initial_average_lambda, adjustment_intervals_3_5) - initial_price if maturity > 3 else 0
    
    cds_price_changes[maturity] = (price_change_0_1, price_change_1_3, price_change_3_5)

# Display the CDS price changes for each scenario and maturity
for maturity, changes in cds_price_changes.items():
    print(f"{maturity}Y CDS Price Changes - [0,1]: {changes[0]:.4f}, [1,3]: {changes[1]:.4f}, [3,5]: {changes[2]:.4f}")


###############################################################################
###############################################################################
###############################################################################

import numpy as np

# Given CVA price changes due to a 10 bps increase in hazard rates
cva_price_changes = np.array([2102.064266053436, 6428.020005798033, 8408.94760371407])  # For intervals [0,1], [1,3], [3,5]

# Given initial and adjusted CDS prices, and their changes for different maturities
cds_price_changes = {
    '1Y': {'[0,1]': -0.0004, '[1,3]': 0.0000, '[3,5]': 0.0000},
    '3Y': {'[0,1]': -0.0011, '[1,3]': -0.0022, '[3,5]': 0.0000},
    '5Y': {'[0,1]': -0.0018, '[1,3]': -0.0036, '[3,5]': -0.0036},
}

# Assuming notional of 1 million for simplicity to calculate impacts
notional = 1_000_000

# Calculate the impact on CVA for each maturity due to a 10 bps increase in hazard rates
cva_impact = {
    '1Y': notional * sum(cds_price_changes['1Y'].values()),
    '3Y': notional * sum(cds_price_changes['3Y'].values()),
    '5Y': notional * sum(cds_price_changes['5Y'].values()),
}

# Determine the notional adjustments needed to neutralize CVA charge movements
# Starting with the 5Y contract and solving backwards
notional_adjustments = {}

# Total impact from initial calculation
total_cva_impact = sum(cva_price_changes)

# Assuming the objective is to neutralize the impact across all intervals
# Start with the 5Y maturity due to its broad impact
notional_adjustments['5Y'] = -total_cva_impact / cva_impact['5Y']

# Adjusting for the 3Y maturity
remaining_impact_after_5y = total_cva_impact + notional_adjustments['5Y'] * cva_impact['5Y']
notional_adjustments['3Y'] = -remaining_impact_after_5y / cva_impact['3Y']

# Finally, adjust the 1Y maturity
remaining_impact_after_3y = remaining_impact_after_5y + notional_adjustments['3Y'] * cva_impact['3Y']
notional_adjustments['1Y'] = -remaining_impact_after_3y / cva_impact['1Y']

print("Notional adjustments needed for CDS contracts to hedge CVA movements:")
for maturity, adjustment in notional_adjustments.items():
    print(f"{maturity}: {adjustment:.2f} million USD")




































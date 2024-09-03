# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:22:39 2024

@author: ieron
"""
# Assuming the EPE array is calculated and available as EPE
collateral_posting_frequencies = [1, 2, 3, 6, 12, 24, 36, 48, 60]  # in months

# Calculate Q values if not already available
# Placeholder: Assuming cumulative_intensity and Q_values calculation here

# Placeholder for recalculating Q_values to match the simulation period
Q_values_monthly = np.repeat(Q_values, 12)[:len(EPE)]  # Adjust if necessary to match EPE length

# Calculate the discount factors for each timestep
discount_factors = np.exp(-r * np.arange(len(EPE)) / 12)

CVA_charges = []  # Store CVA charges for each posting frequency

for freq in collateral_posting_frequencies:
    # Initialize the collateral account and remaining exposure arrays
    collateral_account = np.zeros_like(EPE)
    remaining_exposure = np.zeros_like(EPE)
    
    # Track the total collateral account over time
    for month in range(len(EPE)):
        if month % freq == 0 or freq == 1:  # Post collateral
            collateral_account[month] = EPE[month]
        else:
            collateral_account[month] = collateral_account[month - 1]  # Carry over the previous balance
        
        # Calculate remaining exposure after collateral
        remaining_exposure[month] = max(0, EPE[month] - collateral_account[month])
    
    # Calculate CVA for the adjusted remaining exposure
    adjusted_CVA = np.sum(remaining_exposure[1:] * LGD * discount_factors[1:] * Q_values_monthly)
    CVA_charges.append(adjusted_CVA)

# Plot the impact of the collateral posting frequency on CVA charge
plt.figure(figsize=(10, 6))
plt.plot(collateral_posting_frequencies, CVA_charges, marker='o', linestyle='-')
plt.title('Impact of Collateral Posting Frequency on CVA Charge')
plt.xlabel('Collateral Posting Frequency (months)')
plt.ylabel('CVA Charge')
plt.xticks(collateral_posting_frequencies)
plt.grid(True)
plt.show()
























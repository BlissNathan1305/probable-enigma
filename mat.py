import numpy as np
import pandas as pd

# Define factor ranges (min, max) for each parameter
factors = {
    "Rainfall": (100, 400),          # mm
    "Temperature": (25, 35),         # °C
    "Relative_Humidity": (40, 90),   # %
    "Wind_Speed": (1, 15),           # m/s
    "Pressure": (950, 1050),         # hPa
    "Evaporation": (5, 15)           # mm
}

# Number of experimental runs (adjustable between 50–100)
n_runs = 75

# Generate random samples within ranges
design = np.zeros((n_runs, len(factors)))
for i, (factor, (low, high)) in enumerate(factors.items()):
    design[:, i] = np.random.uniform(low, high, n_runs)

# Simulated crop yield (replace with real model later)
crop_yield = (
    0.002 * design[:, 0] -     # Rainfall effect
    0.05 * design[:, 1] +      # Temperature effect
    0.01 * design[:, 2] -      # Humidity effect
    0.02 * design[:, 3] +      # Wind effect
    0.003 * design[:, 4] -     # Pressure effect
    0.04 * design[:, 5] +      # Evaporation effect
    np.random.normal(0, 0.5, n_runs) # Random noise
)

# Combine into DataFrame
df = pd.DataFrame(design, columns=factors.keys())
df["Crop_Yield"] = crop_yield

# Save to TXT (tab-delimited, MATLAB friendly)
df.to_csv("matlab_experimental_runs.txt", sep="\t", index=False, float_format="%.3f")

print("✅ 75 experimental runs saved to 'matlab_experimental_runs.txt'")

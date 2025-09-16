import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
rainfall = np.array([351.9, 229.7, 201, 192.6, 201, 186, 201, 132.1, 162.9, 116.2])
temperature = np.array([31.2, 30.7, 31.1, 30.8, 31.6, 31.9, 31.9, 31.6, 31.9, 32.5])
crop_yield = np.array([1.78, 2.04, 2.07, 1.94, 1.56, 1.58, 1.66, 1.67, 1.93, 2.05])

# Prepare predictors (X) and response (y)
X = np.column_stack((rainfall, temperature))
y = crop_yield

# Fit linear regression model
model = LinearRegression().fit(X, y)

# Create grid for contour plot
rainfall_grid = np.linspace(rainfall.min(), rainfall.max(), 100)
temperature_grid = np.linspace(temperature.min(), temperature.max(), 100)
R, T = np.meshgrid(rainfall_grid, temperature_grid)

# Predict crop yield on the grid
Z = model.predict(np.column_stack((R.ravel(), T.ravel()))).reshape(R.shape)

# Plot 2D contour
plt.figure(figsize=(8, 6))
contour = plt.contourf(R, T, Z, levels=20, cmap="viridis")
plt.colorbar(contour, label="Predicted Crop Yield")

# Add scatter points for actual data
plt.scatter(rainfall, temperature, c=crop_yield, cmap="viridis", s=100, edgecolors="black", label="Data Points")

plt.xlabel("Rainfall (mm)")
plt.ylabel("Temperature (°C)")
plt.title("2D Contour: Rainfall & Temperature vs Crop Yield")
plt.legend()

# Save plot as JPEG
plt.savefig("contour_rainfall_temp_cropyield.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
plt.show()

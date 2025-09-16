import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ================================
# Insert your data here (10 years)
# ================================
rainfall = np.array([351.9, 229.7, 201, 192.6, 201, 186, 201, 132.1, 162.9, 116.2])  # mm
temperature = np.array([31.2, 30.7, 31.1, 30.8, 31.6, 31.9, 31.9, 31.6, 31.9, 32.5])  # °C
crop_yield = np.array([1.78, 2.04, 2.07, 1.94, 1.56, 1.58, 1.66, 1.67, 1.93, 2.05])   # tonnes/ha

# =================================
# Build Response Surface (2 factors)
# =================================
X = np.column_stack((rainfall, temperature))
y = crop_yield

# Polynomial features for response surface (quadratic)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit regression model
model = LinearRegression()
model.fit(X_poly, y)

# Create a grid for plotting
rainfall_range = np.linspace(min(rainfall), max(rainfall), 50)
temperature_range = np.linspace(min(temperature), max(temperature), 50)
R, T = np.meshgrid(rainfall_range, temperature_range)

X_grid = np.column_stack((R.ravel(), T.ravel()))
X_grid_poly = poly.transform(X_grid)
Y_pred = model.predict(X_grid_poly).reshape(R.shape)

# =======================
# 3D Surface Plot
# =======================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(R, T, Y_pred, cmap='viridis', alpha=0.7)

# Scatter actual data
ax.scatter(rainfall, temperature, crop_yield, color='red', s=50, label="Actual Data")

# Labels
ax.set_xlabel("Rainfall (mm)")
ax.set_ylabel("Temperature (°C)")
ax.set_zlabel("Crop Yield (t/ha)")
ax.set_title("Response Surface: Rainfall & Temperature vs Crop Yield")
ax.legend()

# =======================
# Save instead of show
# =======================
plt.savefig("response_surface.png", dpi=300, bbox_inches="tight")
print("✅ Plot saved as response_surface.png")

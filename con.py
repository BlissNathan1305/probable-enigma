import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Dataset
rainfall = np.array([351.9, 229.7, 201, 192.6, 201, 186, 201, 132.1, 162.9, 116.2])
evaporation = np.array([9.8, 13.8, 9.65, 9.5, 10.5, 10.2, 9.8, 10.85, 10.65, 10.6])
crop_yield = np.array([1.78, 2.04, 2.07, 1.94, 1.56, 1.58, 1.66, 1.67, 1.93, 2.05])

# Prepare input features (Rainfall & Evaporation)
X = np.column_stack((rainfall, evaporation))

# Create polynomial features (degree 2 for smooth surface)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit regression model
model = LinearRegression()
model.fit(X_poly, crop_yield)

# Create grid for plotting
rain_range = np.linspace(min(rainfall), max(rainfall), 100)
evap_range = np.linspace(min(evaporation), max(evaporation), 100)
R, E = np.meshgrid(rain_range, evap_range)

# Predict crop yield on the grid
X_grid = np.column_stack((R.ravel(), E.ravel()))
X_grid_poly = poly.transform(X_grid)
Y_pred = model.predict(X_grid_poly).reshape(R.shape)

# ----- 3D Response Surface Plot -----
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, E, Y_pred, cmap='viridis', alpha=0.8)
ax.scatter(rainfall, evaporation, crop_yield, color='r', s=50, label="Data Points")

ax.set_xlabel("Rainfall (mm)")
ax.set_ylabel("Evaporation (mm)")
ax.set_zlabel("Crop Yield (tons/ha)")
ax.set_title("Response Surface: Rainfall & Evaporation vs Crop Yield")
ax.legend()

plt.savefig("response_surface_rain_evap_crop_yield.jpeg", format="jpeg")
plt.close()

# ----- 2D Contour Plot with Labels -----
plt.figure(figsize=(8, 6))
contour = plt.contourf(R, E, Y_pred, levels=20, cmap='viridis')
plt.scatter(rainfall, evaporation, c=crop_yield, cmap='Reds', edgecolor='k', s=80, label="Data Points")

# Add labels for actual crop yield values
for x, y, z in zip(rainfall, evaporation, crop_yield):
    plt.text(x, y, f"{z:.2f}", fontsize=9, ha='center', va='center', color='white', weight='bold')

plt.colorbar(contour, label="Predicted Crop Yield (tons/ha)")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Evaporation (mm)")
plt.title("Contour Plot: Rainfall & Evaporation vs Crop Yield")
plt.legend()

plt.savefig("contour_rain_evap_crop_yield.jpeg", format="jpeg")
plt.show()

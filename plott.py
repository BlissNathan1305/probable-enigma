import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Data from the user
rainfall = np.array([351.9, 229.7, 201, 192.6, 201, 186, 201, 132.1, 162.9, 116.2])
temperature = np.array([31.2, 30.7, 31.1, 30.8, 31.6, 31.9, 31.9, 31.6, 31.9, 32.5])
crop_yield = np.array([1.78, 2.04, 2.07, 1.94, 1.56, 1.58, 1.66, 1.67, 1.93, 2.05])

# Create mesh grid for response surface
rainfall_range = np.linspace(rainfall.min() - 10, rainfall.max() + 10, 50)
temperature_range = np.linspace(temperature.min() - 0.5, temperature.max() + 0.5, 50)
rainfall_grid, temperature_grid = np.meshgrid(rainfall_range, temperature_range)

# Prepare data for polynomial regression
X = np.column_stack((rainfall, temperature))
y = crop_yield

# Create polynomial regression model (degree 2 for quadratic surface)
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Fit the model
model.fit(X, y)

# Predict values for the grid
X_grid = np.column_stack((rainfall_grid.ravel(), temperature_grid.ravel()))
z_grid = model.predict(X_grid)
z_grid = z_grid.reshape(rainfall_grid.shape)

# Create the 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the response surface
surf = ax.plot_surface(rainfall_grid, temperature_grid, z_grid, 
                      cmap='viridis', alpha=0.8, edgecolor='none')

# Plot the actual data points
scatter = ax.scatter(rainfall, temperature, crop_yield, 
                    c='red', s=100, edgecolor='black', depthshade=True,
                    label='Actual Data Points')

# Customize the plot
ax.set_xlabel('Rainfall (mm)', fontsize=12, labelpad=10)
ax.set_ylabel('Temperature (°C)', fontsize=12, labelpad=10)
ax.set_zlabel('Crop Yield', fontsize=12, labelpad=10)
ax.set_title('Response Surface: Rainfall & Temperature vs Crop Yield', fontsize=14, pad=20)

# Add color bar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
cbar.set_label('Crop Yield', fontsize=10)

# Adjust viewing angle for better perspective
ax.view_init(elev=25, azim=45)

# Add grid and improve layout
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save as JPEG
plt.savefig('response_surface_crop_yield.jpg', dpi=300, bbox_inches='tight', format='jpeg')

# Show the plot
plt.show()

# Print model statistics
print("Response Surface Model Summary:")
print("=" * 40)
print(f"Number of data points: {len(rainfall)}")
print(f"Rainfall range: {rainfall.min():.1f} - {rainfall.max():.1f} mm")
print(f"Temperature range: {temperature.min():.1f} - {temperature.max():.1f} °C")
print(f"Crop Yield range: {crop_yield.min():.2f} - {crop_yield.max():.2f}")
print(f"Model R² score: {model.score(X, y):.3f}")

# Show actual data points
print("\nActual Data Points:")
print("Rainfall (mm)\tTemperature (°C)\tCrop Yield")
print("-" * 45)
for i in range(len(rainfall)):
    print(f"{rainfall[i]:.1f}\t\t{temperature[i]:.1f}\t\t\t{crop_yield[i]:.2f}")

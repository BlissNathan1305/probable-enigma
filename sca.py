import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
rainfall = np.array([351.9, 229.7, 201, 192.6, 201, 186, 201, 132.1, 162.9, 116.2])
temperature = np.array([31.2, 30.7, 31.1, 30.8, 31.6, 31.9, 31.9, 31.6, 31.9, 32.5])
evaporation = np.array([9.8, 13.8, 9.65, 9.5, 10.5, 10.2, 9.8, 10.85, 10.65, 10.6])
crop_yield = np.array([1.78, 2.04, 2.07, 1.94, 1.56, 1.58, 1.66, 1.67, 1.93, 2.05])

# Prepare features (Rainfall, Temperature, Evaporation)
X = np.column_stack((rainfall, temperature, evaporation))
y = crop_yield

# Fit linear regression model
model = LinearRegression().fit(X, y)

# Predicted values
y_pred = model.predict(X)

# Scatter plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color="blue", edgecolors="k", s=100, alpha=0.8)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label="Perfect Prediction")

plt.xlabel("Actual Crop Yield")
plt.ylabel("Predicted Crop Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.legend()
plt.grid(True)

# Save and show
plt.savefig("actual_vs_predicted_crop_yield.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
plt.show()

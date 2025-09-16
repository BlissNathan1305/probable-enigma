import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Create the dataset
data = {
    'RAINFALL': [351.9, 229.7, 201, 192.6, 201, 186, 201, 132.1, 162.9, 116.2],
    'TEMPERATURE': [31.2, 30.7, 31.1, 30.8, 31.6, 31.9, 31.9, 31.6, 31.9, 32.5],
    'CROP_YIELD': [1.78, 2.04, 2.07, 1.94, 1.56, 1.58, 1.66, 1.67, 1.93, 2.05],
    'EVAPORATION': [9.8, 13.8, 9.65, 9.5, 10.5, 10.2, 9.8, 10.85, 10.65, 10.6],
    'RELATIVE_HUMIDITY': [87, 87, 88, 84, 88, 87, 86, 89, 86, 86],
    'WINDSPEED': [3.3, 4, 4.08, 3.75, 4, 3.5, 3.75, 3.58, 4.8, 3.75],
    'PRESSURE': [15.47, 18.1, 17.4, 17.2, 17.1, 16.9, 15.4, 16.6, 17.4, 11.45]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Prepare features and target
X = df.drop('CROP_YIELD', axis=1)
y = df['CROP_YIELD']

# Split the data (with small dataset, we'll use 70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Train model
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'y_pred_test': y_pred_test
    }
    
    print(f"\n{name} Results:")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

# Feature importance for Random Forest
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance)

# Create comprehensive visualization
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Crop Yield Prediction Model Analysis', fontsize=16, fontweight='bold')

# 1. Correlation Heatmap
ax1 = axes[0, 0]
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, ax=ax1, fmt='.2f')
ax1.set_title('Feature Correlation Matrix')

# 2. Feature Importance
ax2 = axes[0, 1]
bars = ax2.bar(range(len(feature_importance)), feature_importance['importance'], 
               color='skyblue', edgecolor='navy', alpha=0.7)
ax2.set_title('Feature Importance (Random Forest)')
ax2.set_xlabel('Features')
ax2.set_ylabel('Importance')
ax2.set_xticks(range(len(feature_importance)))
ax2.set_xticklabels(feature_importance['feature'], rotation=45, ha='right')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 3. Actual vs Predicted (Random Forest)
ax3 = axes[0, 2]
y_pred_rf = results['Random Forest']['y_pred_test']
ax3.scatter(y_test, y_pred_rf, alpha=0.7, color='red', s=100, edgecolor='black')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax3.set_xlabel('Actual Crop Yield')
ax3.set_ylabel('Predicted Crop Yield')
ax3.set_title('Random Forest: Actual vs Predicted')
ax3.grid(True, alpha=0.3)

# Add R² score to the plot
ax3.text(0.05, 0.95, f'R² = {results["Random Forest"]["test_r2"]:.3f}', 
         transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

# 4. Model Performance Comparison
ax4 = axes[1, 0]
model_names = list(results.keys())
r2_scores = [results[name]['test_r2'] for name in model_names]
rmse_scores = [results[name]['test_rmse'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax4.bar(x - width/2, r2_scores, width, label='R² Score', color='lightgreen', alpha=0.8)
ax4_twin = ax4.twinx()
bars2 = ax4_twin.bar(x + width/2, rmse_scores, width, label='RMSE', color='lightcoral', alpha=0.8)

ax4.set_xlabel('Models')
ax4.set_ylabel('R² Score', color='green')
ax4_twin.set_ylabel('RMSE', color='red')
ax4.set_title('Model Performance Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(model_names)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax4.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.01,
             f'{r2_scores[i]:.3f}', ha='center', va='bottom', color='green', fontweight='bold')
    ax4_twin.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.01,
                  f'{rmse_scores[i]:.3f}', ha='center', va='bottom', color='red', fontweight='bold')

# 5. Residuals Analysis
ax5 = axes[1, 1]
residuals = y_test - y_pred_rf
ax5.scatter(y_pred_rf, residuals, alpha=0.7, color='purple', s=100, edgecolor='black')
ax5.axhline(y=0, color='black', linestyle='--', alpha=0.8)
ax5.set_xlabel('Predicted Values')
ax5.set_ylabel('Residuals')
ax5.set_title('Residuals Plot (Random Forest)')
ax5.grid(True, alpha=0.3)

# 6. Feature Distribution
ax6 = axes[1, 2]
# Create a box plot of scaled features to show their distributions
features_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
features_scaled.boxplot(ax=ax6, rot=45)
ax6.set_title('Scaled Feature Distributions')
ax6.set_ylabel('Scaled Values')

plt.tight_layout()

# Save the plot as JPEG
plt.savefig('crop_yield_prediction_results.jpg', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='jpeg')

plt.show()

# Predict crop yield for new data (example)
print("\n" + "="*50)
print("PREDICTION EXAMPLE")
print("="*50)

# Example new data point
new_data = np.array([[200, 31.5, 10.0, 87, 3.8, 16.5]])  # [RAINFALL, TEMP, EVAP, HUMIDITY, WIND, PRESSURE]
new_data_scaled = scaler.transform(new_data)

rf_prediction = rf_model.predict(new_data)
lr_prediction = results['Linear Regression']['model'].predict(new_data_scaled)

print(f"New data point: Rainfall=200, Temp=31.5, Evaporation=10.0, Humidity=87, Wind=3.8, Pressure=16.5")
print(f"Random Forest Prediction: {rf_prediction[0]:.3f}")
print(f"Linear Regression Prediction: {lr_prediction[0]:.3f}")

print("\nModel saved as 'crop_yield_prediction_results.jpg'")
print("Model training completed successfully!")

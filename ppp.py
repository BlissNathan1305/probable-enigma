import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
print(df)
print("\nDataset Statistics:")
print(df.describe())

# Prepare features and target
X = df[['RAINFALL', 'TEMPERATURE', 'EVAPORATION', 'RELATIVE_HUMIDITY', 'WINDSPEED', 'PRESSURE']]
y = df['CROP_YIELD']

print("\nFeature Variables:")
feature_names = list(X.columns)
for i, feature in enumerate(feature_names):
    print(f"X{i+1}: {feature}")

print(f"\nTarget Variable: CROP_YIELD")

# ========================================
# 1. MULTIPLE LINEAR REGRESSION
# ========================================
print("\n" + "="*60)
print("1. MULTIPLE LINEAR REGRESSION MODEL")
print("="*60)

# Fit the model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Get coefficients
intercept = lr_model.intercept_
coefficients = lr_model.coef_

# Create the equation
print("REGRESSION EQUATION:")
equation = f"CROP_YIELD = {intercept:.6f}"
for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
    sign = " + " if coef >= 0 else " - "
    equation += f"{sign}{abs(coef):.6f} × {feature}"

print(equation)

print("\nDETAILED COEFFICIENTS:")
print(f"Intercept (β₀): {intercept:.6f}")
for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
    print(f"β{i+1} ({feature}): {coef:.6f}")

# Model evaluation
y_pred = lr_model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

print(f"\nMODEL PERFORMANCE:")
print(f"R² Score: {r2:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")

# Statistical significance tests
n = len(y)
k = len(coefficients)
residuals = y - y_pred
mse = np.sum(residuals**2) / (n - k - 1)

# Calculate t-statistics and p-values for coefficients
X_with_intercept = np.column_stack([np.ones(n), X])
XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
se_coefficients = np.sqrt(np.diag(XtX_inv * mse))

print(f"\nSTATISTICAL SIGNIFICANCE:")
print(f"{'Parameter':<20} {'Coefficient':<12} {'Std Error':<12} {'t-value':<10} {'p-value':<10}")
print("-" * 70)

# Intercept
t_stat_intercept = intercept / se_coefficients[0]
p_value_intercept = 2 * (1 - stats.t.cdf(abs(t_stat_intercept), n - k - 1))
print(f"{'Intercept':<20} {intercept:<12.6f} {se_coefficients[0]:<12.6f} {t_stat_intercept:<10.4f} {p_value_intercept:<10.4f}")

# Coefficients
for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
    t_stat = coef / se_coefficients[i + 1]
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"{feature:<20} {coef:<12.6f} {se_coefficients[i + 1]:<12.6f} {t_stat:<10.4f} {p_value:<10.4f} {significance}")

# ========================================
# 2. SIMPLIFIED MODELS (STEPWISE APPROACH)
# ========================================
print("\n" + "="*60)
print("2. SIMPLIFIED REGRESSION MODELS")
print("="*60)

# Calculate correlations with target
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print("FEATURE CORRELATIONS WITH CROP YIELD:")
for feature, corr in correlations.items():
    print(f"{feature}: {corr:.4f}")

# Top 3 most correlated features
top_features = correlations.head(3).index.tolist()
print(f"\nTOP 3 FEATURES: {top_features}")

# Simple model with top 3 features
X_simple = X[top_features]
lr_simple = LinearRegression()
lr_simple.fit(X_simple, y)

intercept_simple = lr_simple.intercept_
coef_simple = lr_simple.coef_

print(f"\nSIMPLIFIED MODEL (Top 3 features):")
equation_simple = f"CROP_YIELD = {intercept_simple:.6f}"
for coef, feature in zip(coef_simple, top_features):
    sign = " + " if coef >= 0 else " - "
    equation_simple += f"{sign}{abs(coef):.6f} × {feature}"

print(equation_simple)

y_pred_simple = lr_simple.predict(X_simple)
r2_simple = r2_score(y, y_pred_simple)
print(f"Simplified Model R²: {r2_simple:.6f}")

# ========================================
# 3. POLYNOMIAL REGRESSION
# ========================================
print("\n" + "="*60)
print("3. POLYNOMIAL REGRESSION MODEL (Degree 2)")
print("="*60)

# Use top 2 features for polynomial to avoid overfitting with small dataset
top_2_features = correlations.head(2).index.tolist()
X_poly_base = X[top_2_features]

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_poly_base)
feature_names_poly = poly_features.get_feature_names_out(top_2_features)

lr_poly = LinearRegression()
lr_poly.fit(X_poly, y)

print(f"POLYNOMIAL FEATURES: {list(feature_names_poly)}")
print(f"Intercept: {lr_poly.intercept_:.6f}")
print("Coefficients:")
for i, (coef, fname) in enumerate(zip(lr_poly.coef_, feature_names_poly)):
    print(f"  {fname}: {coef:.6f}")

y_pred_poly = lr_poly.predict(X_poly)
r2_poly = r2_score(y, y_pred_poly)
print(f"Polynomial Model R²: {r2_poly:.6f}")

# ========================================
# 4. REGULARIZED MODELS
# ========================================
print("\n" + "="*60)
print("4. REGULARIZED REGRESSION MODELS")
print("="*60)

# Standardize features for regularization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ridge Regression
ridge = Ridge(alpha=0.1)
ridge.fit(X_scaled, y)
print("RIDGE REGRESSION (α=0.1):")
print(f"Intercept: {ridge.intercept_:.6f}")
for i, (coef, feature) in enumerate(zip(ridge.coef_, feature_names)):
    print(f"  {feature}: {coef:.6f}")

y_pred_ridge = ridge.predict(X_scaled)
r2_ridge = r2_score(y, y_pred_ridge)
print(f"Ridge Model R²: {r2_ridge:.6f}")

# Lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)
print(f"\nLASSO REGRESSION (α=0.01):")
print(f"Intercept: {lasso.intercept_:.6f}")
for i, (coef, feature) in enumerate(zip(lasso.coef_, feature_names)):
    if abs(coef) > 1e-10:  # Only show non-zero coefficients
        print(f"  {feature}: {coef:.6f}")

y_pred_lasso = lasso.predict(X_scaled)
r2_lasso = r2_score(y, y_pred_lasso)
print(f"Lasso Model R²: {r2_lasso:.6f}")

# ========================================
# 5. MODEL COMPARISON AND VISUALIZATION
# ========================================
print("\n" + "="*60)
print("5. MODEL COMPARISON SUMMARY")
print("="*60)

models_comparison = {
    'Multiple Linear Regression': {'R²': r2, 'RMSE': rmse, 'Equation': 'Full model with all features'},
    'Simplified Model': {'R²': r2_simple, 'RMSE': np.sqrt(mean_squared_error(y, y_pred_simple)), 'Equation': 'Top 3 features'},
    'Polynomial Model': {'R²': r2_poly, 'RMSE': np.sqrt(mean_squared_error(y, y_pred_poly)), 'Equation': 'Degree 2 polynomial'},
    'Ridge Regression': {'R²': r2_ridge, 'RMSE': np.sqrt(mean_squared_error(y, y_pred_ridge)), 'Equation': 'L2 regularization'},
    'Lasso Regression': {'R²': r2_lasso, 'RMSE': np.sqrt(mean_squared_error(y, y_pred_lasso)), 'Equation': 'L1 regularization'}
}

print(f"{'Model':<25} {'R²':<10} {'RMSE':<10} {'Description'}")
print("-" * 70)
for model, metrics in models_comparison.items():
    print(f"{model:<25} {metrics['R²']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['Equation']}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Crop Yield Regression Model Analysis', fontsize=16, fontweight='bold')

# 1. Actual vs Predicted (Main model)
ax1 = axes[0, 0]
ax1.scatter(y, y_pred, alpha=0.7, color='blue', s=100, edgecolor='black', label='Predictions')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Crop Yield')
ax1.set_ylabel('Predicted Crop Yield')
ax1.set_title('Multiple Linear Regression: Actual vs Predicted')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
         bbox=dict(boxstyle="round", facecolor='wheat'))

# 2. Feature Coefficients
ax2 = axes[0, 1]
colors = ['red' if coef < 0 else 'green' for coef in coefficients]
bars = ax2.barh(feature_names, coefficients, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Coefficient Value')
ax2.set_title('Feature Coefficients (Multiple Linear Regression)')
ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)

# Add value labels
for i, (bar, coef) in enumerate(zip(bars, coefficients)):
    ax2.text(coef + (0.0001 if coef >= 0 else -0.0001), bar.get_y() + bar.get_height()/2,
             f'{coef:.4f}', ha='left' if coef >= 0 else 'right', va='center', fontweight='bold')

# 3. Model Performance Comparison
ax3 = axes[1, 0]
model_names = list(models_comparison.keys())
r2_scores = [models_comparison[name]['R²'] for name in model_names]
bars = ax3.bar(range(len(model_names)), r2_scores, color='lightblue', alpha=0.8, edgecolor='navy')
ax3.set_xlabel('Models')
ax3.set_ylabel('R² Score')
ax3.set_title('Model Performance Comparison')
ax3.set_xticks(range(len(model_names)))
ax3.set_xticklabels(model_names, rotation=45, ha='right')

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{r2_scores[i]:.3f}', ha='center', va='bottom', fontweight='bold')

# 4. Residuals Analysis
ax4 = axes[1, 1]
residuals = y - y_pred
ax4.scatter(y_pred, residuals, alpha=0.7, color='purple', s=100, edgecolor='black')
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
ax4.set_xlabel('Predicted Values')
ax4.set_ylabel('Residuals')
ax4.set_title('Residuals Plot')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('crop_yield_regression_analysis.jpg', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='jpeg')
plt.show()

# ========================================
# 6. PRACTICAL EQUATION FOR PREDICTION
# ========================================
print("\n" + "="*60)
print("6. FINAL RECOMMENDED EQUATION")
print("="*60)

print("RECOMMENDED EQUATION (Multiple Linear Regression):")
print(equation)
print(f"\nModel Performance: R² = {r2:.4f}, RMSE = {rmse:.4f}")

print("\nPREDICTION EXAMPLE:")
print("For RAINFALL=200, TEMPERATURE=31.5, EVAPORATION=10.0, HUMIDITY=87, WINDSPEED=3.8, PRESSURE=16.5:")

example_data = np.array([[200, 31.5, 10.0, 87, 3.8, 16.5]])
predicted_yield = lr_model.predict(example_data)
print(f"Predicted Crop Yield = {predicted_yield[0]:.4f}")

# Manual calculation verification
manual_calc = (intercept + 
               coefficients[0] * 200 +     # RAINFALL
               coefficients[1] * 31.5 +    # TEMPERATURE  
               coefficients[2] * 10.0 +    # EVAPORATION
               coefficients[3] * 87 +      # RELATIVE_HUMIDITY
               coefficients[4] * 3.8 +     # WINDSPEED
               coefficients[5] * 16.5)     # PRESSURE
print(f"Manual Calculation Verification = {manual_calc:.4f}")

print(f"\nResults saved as 'crop_yield_regression_analysis.jpg'")
print("Analysis completed successfully!")

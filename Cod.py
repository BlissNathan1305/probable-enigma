import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Create the dataset
data = {
    'YEAR': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'RAINFALL': [351.9, 229.7, 201, 192.6, 201, 186, 201, 132.1, 162.9, 116.2],
    'TEMPERATURE': [31.2, 30.7, 31.1, 30.8, 31.6, 31.9, 31.9, 31.6, 31.9, 32.5],
    'EVAPORATION': [9.8, 13.8, 9.65, 9.5, 10.5, 10.2, 9.8, 10.85, 10.65, 10.6],
    'RELATIVE_HUMIDITY': [87, 87, 88, 84, 88, 87, 86, 89, 86, 86],
    'WINDSPEED': [3.3, 4, 4.08, 3.75, 4, 3.5, 3.75, 3.58, 4.8, 3.75],
    'PRESSURE': [15.47, 18.1, 17.4, 17.2, 17.1, 16.9, 15.4, 16.6, 17.4, 11.45],
    'CROP_YIELD': [1.78, 2.04, 2.07, 1.94, 1.56, 1.58, 1.66, 1.67, 1.93, 2.05]
}

# Create DataFrame
df = pd.DataFrame(data)

print("=" * 80)
print("CLIMATE FACTORS SIGNIFICANCE ANALYSIS ON CROP YIELD")
print("=" * 80)
print("Significance Level: α = 0.05")
print("Null Hypothesis (H₀): Climate factor has NO significant effect on crop yield")
print("Alternative Hypothesis (H₁): Climate factor has significant effect on crop yield")
print("=" * 80)

print("\nDATASET OVERVIEW:")
print(df)

print("\nDESCRIPTIVE STATISTICS:")
print(df.describe())

# Prepare data for analysis
climate_factors = ['RAINFALL', 'TEMPERATURE', 'EVAPORATION', 'RELATIVE_HUMIDITY', 'WINDSPEED', 'PRESSURE']
y = df['CROP_YIELD'].values
X = df[climate_factors].values

print("\n" + "=" * 80)
print("1. CORRELATION ANALYSIS")
print("=" * 80)

# Calculate correlations with significance tests
correlations = {}
p_values_corr = {}

for i, factor in enumerate(climate_factors + ['YEAR']):
    if factor == 'YEAR':
        factor_values = df['YEAR'].values
    else:
        factor_values = df[factor].values
    
    corr, p_val = stats.pearsonr(factor_values, y)
    correlations[factor] = corr
    p_values_corr[factor] = p_val

print(f"{'Climate Factor':<20} {'Correlation':<12} {'P-value':<12} {'Significant?':<15}")
print("-" * 65)

significant_factors = []
for factor in climate_factors + ['YEAR']:
    significance = "YES" if p_values_corr[factor] < 0.05 else "NO"
    if p_values_corr[factor] < 0.05:
        significant_factors.append(factor)
    print(f"{factor:<20} {correlations[factor]:<12.4f} {p_values_corr[factor]:<12.4f} {significance:<15}")

print(f"\nSignificant factors at α=0.05: {significant_factors}")

print("\n" + "=" * 80)
print("2. MULTIPLE REGRESSION ANALYSIS")
print("=" * 80)

# Fit multiple regression model using sklearn
model = LinearRegression()
model.fit(X, y)

# Predictions and residuals
y_pred = model.predict(X)
residuals = y - y_pred
n = len(y)
k = len(climate_factors)

# Calculate R-squared and adjusted R-squared
r2 = r2_score(y, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Calculate standard errors and t-statistics manually
mse = np.sum(residuals**2) / (n - k - 1)
X_with_intercept = np.column_stack([np.ones(n), X])
try:
    XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    var_coef = np.diag(XtX_inv) * mse
    se_coef = np.sqrt(var_coef)
    
    # t-statistics
    all_coefs = np.concatenate([[model.intercept_], model.coef_])
    t_stats = all_coefs / se_coef
    
    # p-values (two-tailed test)
    p_values_reg = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
    
    print("REGRESSION COEFFICIENTS AND SIGNIFICANCE:")
    print(f"{'Parameter':<20} {'Coefficient':<12} {'Std Error':<12} {'t-value':<10} {'p-value':<10} {'Significant?'}")
    print("-" * 85)
    
    # Intercept
    sig_intercept = "YES" if p_values_reg[0] < 0.05 else "NO"
    print(f"{'Intercept':<20} {model.intercept_:<12.6f} {se_coef[0]:<12.6f} {t_stats[0]:<10.4f} {p_values_reg[0]:<10.4f} {sig_intercept}")
    
    regression_significant = []
    for i, factor in enumerate(climate_factors):
        significance = "YES" if p_values_reg[i+1] < 0.05 else "NO"
        if p_values_reg[i+1] < 0.05:
            regression_significant.append(factor)
        print(f"{factor:<20} {model.coef_[i]:<12.6f} {se_coef[i+1]:<12.6f} {t_stats[i+1]:<10.4f} {p_values_reg[i+1]:<10.4f} {significance}")
    
    print(f"\nSignificant factors in multiple regression: {regression_significant}")
    
except np.linalg.LinAlgError:
    print("Warning: Matrix is singular. Using correlation analysis results.")
    regression_significant = significant_factors.copy()
    if 'YEAR' in regression_significant:
        regression_significant.remove('YEAR')

print(f"\nModel Performance:")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

# F-test for overall significance
msr = np.sum((y_pred - np.mean(y))**2) / k  # Mean Square Regression
mse_f = np.sum((y - y_pred)**2) / (n - k - 1)  # Mean Square Error
f_stat = msr / mse_f
f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)

print(f"\nOVERALL MODEL F-TEST:")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {f_p_value:.4f}")
print(f"Overall model significant: {'YES' if f_p_value < 0.05 else 'NO'}")

print("\n" + "=" * 80)
print("3. INDIVIDUAL FACTOR SIGNIFICANCE TESTS")
print("=" * 80)

print(f"{'Factor':<20} {'t-statistic':<12} {'P-value':<12} {'R-squared':<12} {'Decision':<25}")
print("-" * 85)

individual_significant = []
individual_results = {}

for i, factor in enumerate(climate_factors):
    # Simple linear regression for each factor
    X_single = df[factor].values.reshape(-1, 1)
    single_model = LinearRegression()
    single_model.fit(X_single, y)
    
    y_pred_single = single_model.predict(X_single)
    r2_single = r2_score(y, y_pred_single)
    
    # Calculate t-statistic for the slope
    residuals_single = y - y_pred_single
    mse_single = np.sum(residuals_single**2) / (n - 2)
    
    # Standard error of the slope
    x_vals = X_single.flatten()
    ssx = np.sum((x_vals - np.mean(x_vals))**2)
    se_slope = np.sqrt(mse_single / ssx)
    
    t_stat = single_model.coef_[0] / se_slope
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    
    decision = "REJECT H₀ (Significant)" if p_val < 0.05 else "FAIL TO REJECT H₀"
    
    individual_results[factor] = {
        't_stat': t_stat,
        'p_value': p_val,
        'r2': r2_single,
        'coefficient': single_model.coef_[0]
    }
    
    if p_val < 0.05:
        individual_significant.append(factor)
    
    print(f"{factor:<20} {t_stat:<12.4f} {p_val:<12.4f} {r2_single:<12.4f} {decision}")

print(f"\nIndividually significant factors: {individual_significant}")

print("\n" + "=" * 80)
print("4. TIME TREND ANALYSIS")
print("=" * 80)

# Test for time trend
time_corr, time_p = stats.pearsonr(df['YEAR'], df['CROP_YIELD'])
print(f"Year vs Crop Yield Correlation: {time_corr:.4f}")
print(f"P-value for time trend: {time_p:.4f}")
print(f"Significant time trend: {'YES' if time_p < 0.05 else 'NO'}")

# Linear trend analysis
X_time = df['YEAR'].values.reshape(-1, 1)
time_model = LinearRegression()
time_model.fit(X_time, y)
time_slope = time_model.coef_[0]

print(f"Time trend slope: {time_slope:.6f} units per year")
trend_direction = "increasing" if time_slope > 0 else "decreasing"
print(f"Crop yield is {trend_direction} over time")

print("\n" + "=" * 80)
print("5. MODEL DIAGNOSTICS")
print("=" * 80)

# Normality test of residuals (Shapiro-Wilk)
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Shapiro-Wilk Normality Test of Residuals:")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4f}")
print(f"  Residuals normally distributed: {'YES' if shapiro_p > 0.05 else 'NO'}")

# Autocorrelation test (Durbin-Watson approximation)
def durbin_watson_stat(residuals):
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)

dw_stat = durbin_watson_stat(residuals)
print(f"\nDurbin-Watson Test (Autocorrelation): {dw_stat:.4f}")
print(f"  Values close to 2.0 indicate no autocorrelation")
print(f"  Autocorrelation concern: {'YES' if dw_stat < 1.5 or dw_stat > 2.5 else 'NO'}")

print("\n" + "=" * 80)
print("6. SUMMARY OF SIGNIFICANT EFFECTS")
print("=" * 80)

# Combine all significant factors
all_significant = set(significant_factors + individual_significant)
if 'regression_significant' in locals():
    all_significant.update(regression_significant)

# Remove YEAR from climate factors summary
climate_significant = [f for f in all_significant if f != 'YEAR']

print("SIGNIFICANT CLIMATE FACTORS (α = 0.05):")
if climate_significant:
    for factor in sorted(climate_significant):
        corr_val = correlations[factor]
        p_val = p_values_corr[factor]
        effect_direction = "Positive" if corr_val > 0 else "Negative"
        effect_strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.4 else "Weak"
        
        print(f"  • {factor}:")
        print(f"    - Correlation: {corr_val:.4f} (p = {p_val:.4f})")
        print(f"    - Effect: {effect_direction} ({effect_strength})")
        if factor in individual_results:
            print(f"    - Individual R²: {individual_results[factor]['r2']:.4f}")
else:
    print("  No climate factors show significant effect at α = 0.05")

print(f"\nOVERALL ASSESSMENT:")
print(f"  • Overall model R²: {r2:.4f} ({r2*100:.1f}% variance explained)")
print(f"  • Model significance: {'SIGNIFICANT' if f_p_value < 0.05 else 'NOT SIGNIFICANT'} (p = {f_p_value:.4f})")
print(f"  • Time trend significance: {'SIGNIFICANT' if time_p < 0.05 else 'NOT SIGNIFICANT'} (p = {time_p:.4f})")
print(f"  • Number of significant climate factors: {len(climate_significant)}")

# Create comprehensive visualization
plt.style.use('default')
fig = plt.figure(figsize=(20, 16))

# Create a custom grid layout
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

# Main title
fig.suptitle('Climate Factors Significance Analysis on Crop Yield (α = 0.05)', 
             fontsize=20, fontweight='bold', y=0.98)

# 1. Correlation heatmap
ax1 = fig.add_subplot(gs[0, :2])
corr_matrix = df[climate_factors + ['CROP_YIELD']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
im = ax1.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

# Add correlation values as text
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        if not mask[i, j]:
            text = ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')

ax1.set_xticks(range(len(corr_matrix.columns)))
ax1.set_yticks(range(len(corr_matrix.columns)))
ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax1.set_yticklabels(corr_matrix.columns)
ax1.set_title('Climate Factors Correlation Matrix', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

# 2. P-values visualization
ax2 = fig.add_subplot(gs[0, 2:])
factors_list = list(climate_factors)
p_vals_list = [p_values_corr[f] for f in factors_list]
colors = ['red' if p < 0.05 else 'lightblue' for p in p_vals_list]

bars = ax2.bar(factors_list, p_vals_list, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=3, label='α = 0.05')
ax2.set_ylabel('P-value', fontweight='bold')
ax2.set_title('Statistical Significance of Climate Factors\n(Red bars: p < 0.05)', fontsize=14, fontweight='bold')
ax2.set_xticklabels(factors_list, rotation=45, ha='right')
ax2.legend()
ax2.set_ylim(0, max(p_vals_list) * 1.1)

# Add p-value labels on bars
for bar, p_val in zip(bars, p_vals_list):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(p_vals_list)*0.02,
             f'{p_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. Individual factor plots (2x3 grid)
for i, factor in enumerate(climate_factors):
    row = 1 + i // 2
    col = (i % 2) * 2
    ax = fig.add_subplot(gs[row, col:col+2])
    
    # Scatter plot with trend line
    x_vals = df[factor].values
    y_vals = df['CROP_YIELD'].values
    
    ax.scatter(x_vals, y_vals, alpha=0.8, s=120, color='blue', edgecolor='black', linewidth=2)
    
    # Add trend line
    z = np.polyfit(x_vals, y_vals, 1)
    p_trend = np.poly1d(z)
    x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_trend, p_trend(x_trend), "r-", alpha=0.8, linewidth=3)
    
    # Correlation and p-value info
    corr = correlations[factor]
    p_val = p_values_corr[factor]
    significance = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
    
    ax.set_xlabel(factor, fontweight='bold')
    ax.set_ylabel('Crop Yield', fontweight='bold')
    ax.set_title(f'{factor} vs Crop Yield\nr = {corr:.3f}, p = {p_val:.3f}\n({significance})', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Color title based on significance
    title_color = 'red' if p_val < 0.05 else 'blue'
    ax.title.set_color(title_color)
    
    # Add R² value
    if factor in individual_results:
        r2_text = f"R² = {individual_results[factor]['r2']:.3f}"
        ax.text(0.05, 0.95, r2_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                fontweight='bold')

# 4. Time series plot
ax_time = fig.add_subplot(gs[3, :2])
years = df['YEAR'].values
yields = df['CROP_YIELD'].values

ax_time.plot(years, yields, 'bo-', linewidth=3, markersize=10, 
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=2)
ax_time.set_xlabel('Year', fontweight='bold')
ax_time.set_ylabel('Crop Yield', fontweight='bold')

time_significance = "SIGNIFICANT" if time_p < 0.05 else "NOT SIGNIFICANT"
ax_time.set_title(f'Crop Yield Time Trend (2011-2020)\nr = {time_corr:.3f}, p = {time_p:.3f} ({time_significance})', 
                 fontsize=12, fontweight='bold')
ax_time.grid(True, alpha=0.3)

# Add trend line for time series
z_time = np.polyfit(years, yields, 1)
p_time = np.poly1d(z_time)
ax_time.plot(years, p_time(years), "r--", alpha=0.8, linewidth=3)

# Color title based on significance
title_color = 'red' if time_p < 0.05 else 'blue'
ax_time.title.set_color(title_color)

# 5. Model diagnostics plot
ax_diag = fig.add_subplot(gs[3, 2:])

# Residuals vs fitted plot
fitted_vals = y_pred
ax_diag.scatter(fitted_vals, residuals, alpha=0.8, s=120, color='purple', 
                edgecolor='black', linewidth=2)
ax_diag.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
ax_diag.set_xlabel('Fitted Values', fontweight='bold')
ax_diag.set_ylabel('Residuals', fontweight='bold')
ax_diag.set_title(f'Residuals vs Fitted Values\nNormality: p = {shapiro_p:.3f}, DW = {dw_stat:.3f}', 
                 fontsize=12, fontweight='bold')
ax_diag.grid(True, alpha=0.3)

# Add reference lines
ax_diag.axhline(y=np.std(residuals), color='orange', linestyle=':', alpha=0.6, label='+1 SD')
ax_diag.axhline(y=-np.std(residuals), color='orange', linestyle=':', alpha=0.6, label='-1 SD')
ax_diag.legend()

# Save the plot
plt.savefig('climate_factors_significance_analysis.jpg', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', format='jpeg')
plt.show()

# Final comprehensive summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY TABLE")
print("=" * 80)

summary_data = []
for factor in climate_factors:
    corr_val = correlations[factor]
    p_val = p_values_corr[factor]
    is_significant = "YES" if p_val < 0.05 else "NO"
    effect_direction = "Positive" if corr_val > 0 else "Negative"
    
    if abs(corr_val) > 0.7:
        effect_strength = "Strong"
    elif abs(corr_val) > 0.4:
        effect_strength = "Moderate"
    else:
        effect_strength = "Weak"
    
    summary_data.append([
        factor,
        f"{corr_val:.4f}",
        f"{p_val:.4f}",
        is_significant,
        effect_direction,
        effect_strength
    ])

# Print summary table
print(f"{'Factor':<20} {'Correlation':<12} {'P-value':<12} {'Significant':<12} {'Direction':<12} {'Strength'}")
print("-" * 95)
for row in summary_data:
    print(f"{row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]}")

print(f"\n{'='*80}")
print("STATISTICAL CONCLUSIONS")
print("="*80)

if climate_significant:
    print(f"✓ SIGNIFICANT CLIMATE FACTORS (α = 0.05): {len(climate_significant)} factors")
    for factor in sorted(climate_significant):
        effect = "increases" if correlations[factor] > 0 else "decreases"
        strength = "strong" if abs(correlations[factor]) > 0.7 else "moderate" if abs(correlations[factor]) > 0.4 else "weak"
        print(f"  • {factor}: {effect} crop yield ({strength} effect, r={correlations[factor]:.3f})")
else:
    print("✗ NO SIGNIFICANT CLIMATE FACTORS at α = 0.05")

print(f"\n✓ OVERALL MODEL:")
print(f"  • Explains {r2*100:.1f}% of crop yield variance")
print(f"  • Statistical significance: {'SIGNIFICANT' if f_p_value < 0.05 else 'NOT SIGNIFICANT'} (p = {f_p_value:.4f})")

print(f"\n✓ TIME TREND (2011-2020):")
print(f"  • Trend: {'SIGNIFICANT' if time_p < 0.05 else 'NOT SIGNIFICANT'} (p = {time_p:.4f})")
if time_p < 0.05:
    trend_desc = "increasing" if time_slope > 0 else "decreasing"
    print(f"  • Crop yield is {trend_desc} at {abs(time_slope):.4f} units per year")

print(f"\nAnalysis completed! Results saved as 'climate_factors_significance_analysis.jpg'")
print("="*80)

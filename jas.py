# ------------------------------------------------------------
# Quick climate-change proxy model
# Predicts NEXT-YEAR mean annual temperature from past
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import warnings, os, sys
warnings.filterwarnings("ignore")

# 1. Load tidy data ----------------------------------------------------------
df = pd.read_csv("tidy_Edata.csv")

# 2. Build annual aggregates -------------------------------------------------
annual = (df.assign(year=lambda x: x["year"].astype(int),
                    month=lambda x: pd.to_datetime(x["month"], format="%b").dt.month)
            .pivot_table(index=["year", "month"],
                         columns="variable",
                         values="value")
            .reset_index()
            .groupby("year")
            .agg({"Evaporation": "mean",
                  "Wind": "mean",
                  "UYO":  ["mean", "max", "min"],
                  "RAIN": "sum",
                  "PRESSURE": "mean",
                  "HUMIDITY": "mean"})
            .round(2))

# flatten multi-index columns
annual.columns = ["_".join(col).strip() if col[1] else col[0]
                  for col in annual.columns.values]
annual = annual.reset_index()

# 3. Create lagged features & target -----------------------------------------
LAGS = 3
for col in annual.columns:
    if col=="year": continue
    for lag in range(1,LAGS+1):
        annual[f"{col}_lag{lag}"] = annual[col].shift(lag)

# target = next-year mean temperature
annual["target"] = annual["UYO_mean"].shift(-1)

# drop rows with NaNs introduced by lagging / target
annual = annual.dropna()

# 4. Train / test split ------------------------------------------------------
X = annual.drop(columns=["year","target"])
y = annual["target"]
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, shuffle=False)   # time-series → keep order

# 5. Model zoo ---------------------------------------------------------------
models = {
    "LR":  LinearRegression(),
    "RF":  RandomForestRegressor(n_estimators=300, random_state=42),
    "XGB": xgb.XGBRegressor(objective="reg:squarederror",
                            n_estimators=300, learning_rate=.05,
                            max_depth=4, subsample=.8, colsample_bytree=.8,
                            random_state=42)
}

best_model, best_score = None, 1e9
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2  = r2_score(y_test, pred)
    print(f"{name:3} | MAE={mae:5.2f} °C | R²={r2:5.3f}")
    if mae < best_score:
        best_model, best_score = model, mae

# 6. Persist winner ----------------------------------------------------------
joblib.dump(best_model, "uyo_temp_predictor.pkl")
print("\nBest model saved → uyo_temp_predictor.pkl")

# 7. Quick sanity plot -------------------------------------------------------
import matplotlib.pyplot as plt
pred_best = best_model.predict(X_test)
plt.figure(figsize=(8,4))
plt.plot(annual["year"].iloc[-len(y_test):], y_test, label="observed")
plt.plot(annual["year"].iloc[-len(y_test):], pred_best, label="predicted")
plt.title("Uyo mean annual temperature – forecast vs actual")
plt.legend(); plt.tight_layout(); plt.savefig("uyo_forecast.png")


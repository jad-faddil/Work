import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# Load
data = pd.read_parquet("Data/data_weekly.parquet")

for col in ["sex", "age_group", "departement"]:
    data[col] = data[col].astype("category")
    
# Split
train = data[data["year"] <= 2020]
valid = data[data["year"].between(2021, 2022)]
test = data[data["year"].between(2023, 2024)]

# Features
features_base = ["sex", "age_group", "departement", "year", "exposure"]
features_temp = features_base + ["TN", "TM", "TX", "TX_heat", "TN_heat", "heatdays_count"]

X_train_base, X_valid_base, X_test_base = train[features_base], valid[features_base], test[features_base]
X_train_temp, X_valid_temp, X_test_temp = train[features_temp], valid[features_temp], test[features_temp]

y_train = train["deaths"]
y_valid = valid["deaths"]

# Predictions
model_base = xgb.XGBRegressor()
model_base.load_model("Results/Python_results/model_weekly_base.json")

model_temp = xgb.XGBRegressor()
model_temp.load_model("Results/Python_results/model_weekly_temp.json")

pred_base = model_base.predict(X_test_base)
pred_temp = model_temp.predict(X_test_temp)

# Poisson deviance
def poisson_deviance(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
    dev = np.zeros_like(y)
    positive = y > 0
    dev[positive] = 2 * (y[positive] * np.log(y[positive] / mu[positive]) - (y[positive] - mu[positive]))
    dev[~positive] = 2 * mu[~positive]
    return dev.sum()

y_test = test["deaths"].to_numpy()

# Exposure-adjusted null model
null_rate = train["deaths"].sum() / train["exposure"].sum()
null_pred = null_rate * test["exposure"].to_numpy()

dev_null = poisson_deviance(y_test, null_pred)
dev_base = poisson_deviance(y_test, pred_base)
dev_temp = poisson_deviance(y_test, pred_temp)

print("\n===== MODEL EVALUATION =====")
print(f"Null deviance:          {dev_null:,.2f}")
print(f"Baseline deviance:      {dev_base:,.2f}")
print(f"Temperature deviance:   {dev_temp:,.2f}")
print(f"\nBaseline / null:         {dev_base / dev_null:.4f}")
print(f"Temperature / null:      {dev_temp / dev_null:.4f}")
print(f"\nBaseline improvement:    {1 - dev_base / dev_null:.2%}")
print(f"Temperature improvement: {1 - dev_temp / dev_null:.2%}")
print(f"Temperature vs baseline: {(dev_base - dev_temp) / dev_base:.2%}")

print("\n===== Yearly deviance =====")
for year in [2023, 2024]:
    d = test[test["year"] == year]
    pred_base_year = model_base.predict(d[features_base])
    pred_temp_year = model_temp.predict(d[features_temp])
    print(f"{year} : {poisson_deviance(d["deaths"], pred_base_year):.0f}  {poisson_deviance(d["deaths"], pred_temp_year):.0f}")
    
# Daily national predictions
results = test[["date", "year", "deaths"]].copy()
results["pred_base"] = pred_base
results["pred_temp"] = pred_temp

daily = results.groupby(["date", "year"])[["deaths", "pred_base", "pred_temp"]].sum().reset_index()
daily["error_base"] = daily["pred_base"] - daily["deaths"]
daily["error_temp"] = daily["pred_temp"] - daily["deaths"]

daily["abs_error_base"] = daily["error_base"].abs()
daily["abs_error_temp"] = daily["error_temp"].abs()

daily["sq_error_base"] = daily["error_base"] ** 2
daily["sq_error_temp"] = daily["error_temp"] ** 2

# Prediction sanity check
print("\n===== PREDICTION CHECK =====")
print(f"Observed total:     {y_test.sum():,.0f}")
print(f"Baseline total:     {pred_base.sum():,.0f}")
print(f"Temperature total:  {pred_temp.sum():,.0f}")
for model in ["pred_base", "pred_temp"]:
    error = daily[model] - daily["deaths"]
    print(f"\n{model}")
    print(f"ME:   {error.mean():.2f}")
    print(f"MAE:  {error.abs().mean():.2f}")
    print(f"MSE:  {(error ** 2).mean():.2f}")

for year in [2023, 2024]:
    d = daily[daily["year"] == year]

    print(f"\n===== {year} =====")
    print("Baseline MAE:", d["abs_error_base"].mean())
    print("Temperature MAE:", d["abs_error_temp"].mean())
    print("Baseline MSE:", d["sq_error_base"].mean())
    print("Temperature MSE:", d["sq_error_temp"].mean())

# Plots
for year in [2023, 2024]:
    yearly = daily[daily["year"] == year]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(yearly["date"], yearly["deaths"], label="Observed")
    ax.plot(yearly["date"], yearly["pred_base"], label="Baseline")
    ax.plot(yearly["date"], yearly["pred_temp"], label="Temperature")

    ax.set_title(f"Observed vs predicted deaths — {year}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Deaths")
    ax.legend()
    plt.tight_layout()
    plt.show()
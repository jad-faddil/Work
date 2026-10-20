import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

# Loading and thinning data
print("Loading data...")
data = pd.read_parquet("Data/data_weekly.parquet")
data.rename(columns={"week": "date"}, inplace=True)
data["year"] = data["date"].dt.year
data["mortality_rate"] = data["deaths"] / data["exposure"]

week_of_year = data["date"].dt.isocalendar().week.astype(int)
data["sin_week"] = np.sin(2 * np.pi * week_of_year / 52)
data["cos_week"] = np.cos(2 * np.pi * week_of_year / 52)

groups = ["departement", "sex", "age_group"]
for i in range(1,6):
    data[f"rate_lag{i}"] = data.groupby(groups)["mortality_rate"].shift(i)

train = data[(data["date"] >= "2004-01-01") & (data["date"] <= "2022-12-31")].copy()
test = data[data["date"] >= "2023-01-01"].copy()

temp_cols = ["TN_mean", "TN_hot_days", 
             "TX_mean", "TX_hot_days"]

def fit_glm(cols):
    tr = train.dropna(subset=cols).copy()
    te = test.dropna(subset=cols).copy()

    categorical = ["sex", "age_group"]
    numeric = ["departement", "year", "sin_week", "cos_week"] + cols

    X_train = pd.get_dummies(tr[categorical], drop_first=True)
    X_test = pd.get_dummies(te[categorical], drop_first=True)

    X_train[numeric] = tr[numeric].values
    X_test[numeric] = te[numeric].values

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_train = sm.add_constant(X_train).astype(float)
    X_test = sm.add_constant(X_test).astype(float)

    model = sm.GLM(tr["deaths"], X_train, family=sm.families.Poisson(), offset=np.log(tr["exposure"])).fit()
    pred = model.predict(X_test, offset=np.log(te["exposure"]))
    return model, te, pred

experiments = {
    "Baseline": [],
    "Temperature": temp_cols,
    "Lags 1-5" : temp_cols + [f"rate_lag{i}" for i in range(1,6)]}

results = []

for name, cols in experiments.items():
    model, te, pred = fit_glm(cols)

    y = te["deaths"] / te["exposure"]
    pred_rate = pred / te["exposure"]

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y, pred_rate),
        "RMSE": np.sqrt(mean_squared_error(y, pred_rate)),
        "R2": 1 - np.sum((y - pred_rate) ** 2) / np.sum((y - y.mean()) ** 2)
    })

results = pd.DataFrame(results)
print(results.to_string(index=False))

predictions = {}

for name, cols in experiments.items():
    model, te, pred = fit_glm(cols)

    te = te.copy()
    te["prediction"] = pred
    predictions[name] = te.groupby("date").agg(deaths=("deaths", "sum"), prediction=("prediction", "sum"))

    predictions[name][f"lower"] = poisson.ppf(0.025, predictions[name]["prediction"])
    predictions[name][f"upper"] = poisson.ppf(0.975, predictions[name]["prediction"])

for name in experiments.keys():
    pred = predictions[name]
    inside = ((pred["deaths"] >= pred["lower"]) & (pred["deaths"] <= pred["upper"]))

    print(f"{name}: {inside.mean() * 100:.1f}% of points inside 95% interval")

for i in range(1, 6):
    print(f"lag{i}: coef={model.params[f'rate_lag{i}']:.6f}, p={model.pvalues[f'rate_lag{i}']}")

# Plot selected models
for year in [2023, 2024]:
    fig, ax = plt.subplots(figsize=(14, 5))

    obs = predictions["Baseline"]
    obs = obs[obs.index.year == year]

    ax.plot(obs.index, obs["deaths"], label="Observed", linewidth=2)

    for name in experiments.keys():
        pred = predictions[name]
        pred = pred[pred.index.year == year]
        ax.plot(pred.index, pred["prediction"], label=name, alpha=0.8)
        ax.fill_between(pred.index, pred[f"lower"], pred[f"upper"], alpha=0.12)

    ax.set_title(f"Weekly deaths — {year}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Deaths")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
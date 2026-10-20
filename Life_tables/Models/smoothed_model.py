import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error

population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_all.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")
deaths = deaths.groupby(["Year", "Week", "Age"], as_index=False)["Deaths"].sum()

data = population.merge(deaths, on=["Year", "Week", "Age"], how="left")
data["Deaths"] = data["Deaths"].fillna(0).astype(int)
data["Population"] = data["Total"].round().astype(int)
data = data[data["Week"].between(18, 44) & data["Age"].between(20, 100) & data["Year"].between(2015, 2025) & (data["Population"] > 0)].copy()

def fit_baseline(train):
    b = train.groupby("Age").agg(D=("Deaths", "sum"), N=("Population", "sum"))
    return b["D"] / b["N"]

def fit_seasonal(train, qbase, df=None):
    train = train.copy()
    train["qbase"] = train["Age"].map(qbase)
    formula = "Deaths ~ C(Age) * C(Week)" if df is None else (
        f"Deaths ~ 0 + C(Age):bs(Week, df={df}, degree=3, include_intercept=False, lower_bound=18, upper_bound=44)"
        )
    return smf.glm(formula, data=train, family=sm.families.Poisson(), offset=np.log(train["Population"] * train["qbase"])).fit()

def predict_qx(model, frame, qbase, weeks=range(18, 45)):
    frame = frame.copy()
    frame["qbase"] = frame["Age"].map(qbase)
    frame["qx_pred"] = model.predict(frame, offset=np.log(frame["Population"] * frame["qbase"])) / frame["Population"]

    # Center the seasonal effect so the geometric mean multiplier is 1
    ages = sorted(frame["Age"].unique())
    grid = pd.DataFrame([(a, w, 1) for a in ages for w in weeks], columns=["Age", "Week", "Population"])
    grid["qbase"] = grid["Age"].map(qbase)
    grid["rate"] = model.predict(grid, offset=np.log(grid["qbase"]))
    grid["log_mult"] = np.log(np.maximum(grid["rate"], 1e-15) / grid["qbase"])
    center = grid.groupby("Age")["log_mult"].mean()
    frame["qx_pred"] /= np.exp(frame["Age"].map(center))
    return frame["qx_pred"]

def scores(actual, predicted):
    return (mean_absolute_error(actual, predicted), np.sqrt(mean_squared_error(actual, predicted)))

# Validation: fit on 2018–2021, choose spline flexibility on 2022–2023
fit = data[data["Year"].between(2015, 2021)].copy()
valid = data[data["Year"].between(2022, 2023)].copy()
qbase_val = fit_baseline(fit)
valid["qx"] = valid["Deaths"] / valid["Population"]

best_df = 6

# Final fit: all 2018–2023, evaluate on 2024–2025
train = data[data["Year"].between(2015, 2023)].copy()
test = data[data["Year"].between(2024, 2025)].copy()
qbase = fit_baseline(train)

test["qx_observed"] = test["Deaths"] / test["Population"]
test["qx_baseline"] = test["Age"].map(qbase)

# Original categorical-week model
categorical_model = fit_seasonal(train, qbase)
test["qx_categorical"] = predict_qx(categorical_model, test, qbase)

# Selected smoothed model
smooth_model = fit_seasonal(train, qbase, df=best_df)
test["qx_smoothed"] = predict_qx(smooth_model, test, qbase)

for name, col in [("Baseline", "qx_baseline"), ("Categorical", "qx_categorical"), ("Smoothed", "qx_smoothed") ]:
    mae, rmse = scores(test["qx_observed"], test[col])
    print(f"{name}: MAE={mae:.8g}, RMSE={rmse:.8g}")

# test.to_parquet("qx_predictions_smoothed_2024_2025.parquet", index=False)

weeks_to_compare = [(2024, 20), (2024, 30), (2024, 40),
                    (2025, 20), (2025, 30), (2025, 40)]

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)

for ax, (year, week) in zip(axes.flat, weeks_to_compare):
    df = test[(test["Year"] == year) & (test["Week"] == week)].sort_values("Age")

    ax.plot(df["Age"], df["qx_observed"], label="Observed", marker=".", markersize=3)
    ax.plot(df["Age"], df["qx_baseline"], label="Baseline", linestyle="--")
    ax.plot(df["Age"], df["qx_categorical"], label="Categorical seasonal", linestyle=":")
    ax.plot(df["Age"], df["qx_smoothed"], label="Smoothed seasonal", linestyle="-.")

    ax.set_yscale("log")
    ax.set_title(f"{year}, week {week}")
    ax.set_xlabel("Age")
    ax.grid(alpha=0.3, which="both")

axes[0, 0].set_ylabel("Weekly mortality probability ($q_x$)")
axes[1, 0].set_ylabel("Weekly mortality probability ($q_x$)")
axes[0, 0].legend()

plt.tight_layout()
plt.show()

ages = [30, 50, 70, 85, 95]
weeks = range(18, 45)

grid = pd.DataFrame([(age, week) for age in ages for week in weeks], columns=["Age", "Week"])
grid["Population"] = 1
grid["qbase"] = grid["Age"].map(qbase)

# Categorical model multiplier
grid["categorical_rate"] = categorical_model.predict(grid, offset=np.log(grid["qbase"]))
grid["categorical_multiplier"] = grid["categorical_rate"] / grid["qbase"]

# Smoothed model multiplier
grid["smoothed_rate"] = smooth_model.predict(grid, offset=np.log(grid["qbase"]))
grid["smoothed_multiplier"] = grid["smoothed_rate"] / grid["qbase"]

plt.figure(figsize=(11, 6))

for age in ages:
    df = grid[grid["Age"] == age]
    #plt.plot(df["Week"], df["categorical_multiplier"], marker=".", markersize=3, alpha=0.55, label=f"Age {age} categorical")
    plt.plot(df["Week"], df["smoothed_multiplier"], linewidth=2, label=f"Age {age} smoothed")

plt.axhline(1, color="black", linestyle="--", linewidth=1)
plt.xlabel("Week of year")
plt.ylabel("Seasonal multiplier ($\\hat q_x / q_x^{base}$)")
plt.title("Categorical vs smoothed seasonal effects by age")
plt.xticks(list(weeks))
plt.grid(alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
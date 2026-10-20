import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_all.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")

deaths = deaths.groupby(["Year", "Week", "Age"], as_index=False)["Deaths"].sum()

data = population.merge(deaths, on=["Year", "Week", "Age"], how="left")
data["Deaths"] = data["Deaths"].fillna(0).astype(int)
data["Population"] = data["Total"].round().astype(int)

data = data[
    data["Week"].between(18, 44)
    & data["Age"].between(20, 100)
    & data["Year"].between(2018, 2025)
    & (data["Population"] > 0)
].copy()

train = data[data["Year"].between(2018, 2023)].copy()
test = data[data["Year"].between(2024, 2025)].copy()

# Baseline qx by age
baseline = train.groupby("Age").agg(Deaths=("Deaths", "sum"), Population=("Population", "sum"))
baseline["qx_baseline"] = baseline["Deaths"] / baseline["Population"]

train = train.join(baseline["qx_baseline"], on="Age")
test = test.join(baseline["qx_baseline"], on="Age")

# Learn age-specific weekly adjustments relative to baseline
model = smf.glm("Deaths ~ C(Age) * C(Week)", data=train, family=sm.families.Poisson(), offset=np.log(train["Population"] * train["qx_baseline"])).fit()

# Predict the adjustment to baseline qx
test["predicted_deaths"] = model.predict(test, offset=np.log(test["Population"] * test["qx_baseline"]))
test["qx_predicted"] = test["predicted_deaths"] / test["Population"]
test["qx_observed"] = test["Deaths"] / test["Population"]

# Baseline prediction for comparison
test["qx_baseline_prediction"] = test["qx_baseline"]

print("Baseline MAE:", mean_absolute_error(test["qx_observed"], test["qx_baseline_prediction"]))
print("Seasonal model MAE:", mean_absolute_error(test["qx_observed"], test["qx_predicted"]))

print("Baseline RMSE:", root_mean_squared_error(test["qx_observed"], test["qx_baseline_prediction"]))
print("Seasonal model RMSE:", root_mean_squared_error(test["qx_observed"], test["qx_predicted"]))

print("Model parameters:", len(model.params))
print("Training rows:", len(train), "| Test rows:", len(test))

weeks_to_compare = [(2024, 20), (2024, 30), (2024, 40),
                    (2025, 20), (2025, 30), (2025, 40)]

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)

for ax, (year, week) in zip(axes.flat, weeks_to_compare):
    df = test[(test["Year"] == year) & (test["Week"] == week)].sort_values("Age")

    ax.plot(df["Age"], df["qx_observed"], label="Observed", marker=".", markersize=3)
    ax.plot(df["Age"], df["qx_baseline_prediction"], label="Baseline", linestyle="--")
    ax.plot(df["Age"], df["qx_predicted"], label="Seasonal model", linestyle=":")

    ax.set_yscale("log")
    ax.set_title(f"{year}, week {week}")
    ax.set_xlabel("Age")
    ax.grid(alpha=0.3)

axes[0, 0].set_ylabel("Weekly mortality probability ($q_x$)")
axes[1, 0].set_ylabel("Weekly mortality probability ($q_x$)")
axes[0, 0].legend()

plt.tight_layout()
plt.show()

ages = [30, 50, 70, 85, 95]
weeks = range(18, 45)

grid = pd.DataFrame([(age, week) for age in ages for week in weeks], columns=["Age", "Week"])

# Predict mortality rates using the fitted seasonal model
grid["Population"] = 1
grid["qx_baseline"] = grid["Age"].map(baseline["qx_baseline"])

grid["predicted_deaths"] = model.predict(grid, offset=np.log(grid["Population"] * grid["qx_baseline"]))

grid["qx_predicted"] = grid["predicted_deaths"] / grid["Population"]
grid["multiplier"] = grid["qx_predicted"] / grid["qx_baseline"]

plt.figure(figsize=(11, 6))

for age in ages:
    df = grid[grid["Age"] == age]
    plt.plot(df["Week"], df["multiplier"], marker="o", markersize=3, label=f"Age {age}")

plt.axhline(1, color="black", linestyle="--", linewidth=1)
plt.xlabel("Week of year")
plt.ylabel("Seasonal multiplier (predicted qx / baseline qx)")
plt.title("Learned seasonal mortality effect by age")
plt.xticks(list(weeks))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
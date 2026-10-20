import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from patsy import bs, dmatrix, build_design_matrices

population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_all.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")
weather = pd.read_parquet("Life_tables/Life_tables_data/weekly_weather.parquet")

deaths = deaths.groupby(["Year", "Week", "Age"], as_index=False)["Deaths"].sum()
weather = weather.rename(columns={"year": "Year", "week": "Week"})
weather_cols = ["TN", "TX", "UN", "UX"]
weather = weather.groupby(["Year", "Week"])[weather_cols].mean().reset_index()

data = population.merge(deaths, on=["Year", "Week", "Age"], how="left")
data["Deaths"] = data["Deaths"].fillna(0).astype(int)
data["Population"] = data["Total"].round().astype(int)
data = data[data["Age"].between(20, 85) & data["Week"].between(18, 44)].merge(weather, on=["Year", "Week"], how="left")

# Baseline qx
qbase = data[data["Year"].between(2019, 2023)].groupby("Age").apply(lambda g: g["Deaths"].sum() / g["Population"].sum(), include_groups=False)

# Seasonal model: shared effect across ages
train = data[data["Year"].between(2014, 2023)].copy()
train["qbase"] = train["Age"].map(qbase)

seasonal_model = smf.glm("Deaths ~ bs(Week, df=6, degree=3, include_intercept=False, lower_bound=18, upper_bound=44)", data=train, family=sm.families.Poisson(), offset=np.log(train["Population"] * train["qbase"])).fit()

# Normalize seasonal multiplier to mean 1
weeks = pd.DataFrame({"Week": np.arange(18, 45)})
X_weeks = dmatrix("bs(Week, df=6, degree=3, include_intercept=False, lower_bound=18, upper_bound=44)", weeks, return_type="dataframe")
seasonal_multiplier = pd.Series(np.exp(np.asarray(X_weeks) @ seasonal_model.params), index=weeks["Week"])
seasonal_multiplier /= seasonal_multiplier.mean()

# Weather correction
train["seasonal_multiplier"] = train["Week"].map(seasonal_multiplier)
train["qx_smoothed"] = train["qbase"] * train["seasonal_multiplier"]
weather_train = train.dropna(subset=weather_cols)

weather_model = smf.glm(f"Deaths ~ {' + '.join(weather_cols)}", data=weather_train, family=sm.families.Poisson(), offset=np.log(weather_train["Population"] * weather_train["qx_smoothed"])).fit()

# Test
test = data[data["Year"].between(2024, 2025)].copy()
test["qx_observed"] = test["Deaths"] / test["Population"]
test["qx_baseline"] = test["Age"].map(qbase)
test["seasonal_multiplier"] = test["Week"].map(seasonal_multiplier)
test["qx_smoothed"] = test["qx_baseline"] * test["seasonal_multiplier"]
test["qx_weather"] = weather_model.predict(test, offset=np.log(test["Population"] * test["qx_smoothed"])) / test["Population"]


############################### Metrics ###############################

models = {"Baseline": "qx_baseline", "Smoothed seasonal": "qx_smoothed", "Seasonal + weather": "qx_weather"}

metrics = pd.DataFrame([
    {
    "Model": name, 
    "MAE": mean_absolute_error(test["qx_observed"], test[col]), 
    "RMSE": np.sqrt(mean_squared_error(test["qx_observed"], test[col])), 
    "WAPE (%)": 100 * (test[col] - test["qx_observed"]).abs().sum() / test["qx_observed"].abs().sum(),
    "Mean relative bias (%)": 100 * (test[col] - test["qx_observed"]).sum() / test["qx_observed"].abs().sum()
    } 
    for name, col in models.items()])

metrics["MAE improvement (%)"] = (1 - metrics["MAE"] / metrics.loc[0, "MAE"]) * 100
metrics["RMSE improvement (%)"] = (1 - metrics["RMSE"] / metrics.loc[0, "RMSE"]) * 100

print("\nModel comparison:")
print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nWeather coefficients:")
print(weather_model.params)
print(weather_model.pvalues)
print(data[weather_cols].corr().round(3))

################################ Plots ################################

# Plot 1
weeks_to_compare = [(2024, 20), (2024, 30), (2024, 40),
                    (2025, 20), (2025, 30), (2025, 40)]
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)

for ax, (year, week) in zip(axes.flat, weeks_to_compare):
    df = test[(test["Year"] == year) & (test["Week"] == week)].sort_values("Age")
    ax.plot(df["Age"], df["qx_observed"], label="Observed", marker=".", markersize=3)
    ax.plot(df["Age"], df["qx_baseline"], label="Baseline", linestyle="--")
    ax.plot(df["Age"], df["qx_smoothed"], label="Shared seasonality", linestyle="-.")
    ax.plot(df["Age"], df["qx_weather"], label="Seasonality + weather")
    ax.set(yscale="log", title=f"{year}, week {week}", xlabel="Age")
    ax.grid(alpha=0.3, which="both")

axes[0, 0].set_ylabel("Weekly mortality probability ($q_x$)")
axes[1, 0].set_ylabel("Weekly mortality probability ($q_x$)")
axes[0, 0].legend()
plt.tight_layout()
plt.show()

# Plot 2: shared seasonal multiplier
plt.figure(figsize=(11, 6))
plt.plot(seasonal_multiplier.index, seasonal_multiplier.values, linewidth=2, label="Shared seasonal multiplier")
plt.axhline(1, color="black", linestyle="--", linewidth=1)
plt.xlabel("Week of year")
plt.ylabel("Seasonal multiplier")
plt.title("Shared seasonal effect across ages")
plt.xticks(list(range(18, 45)))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: weather multiplier
weather_test = data.loc[data["Year"].between(2024, 2025), ["Year", "Week"] + weather_cols].drop_duplicates(["Year", "Week"]).dropna(subset=weather_cols).copy()
weather_test["weather_multiplier"] = np.exp(weather_model.params["Intercept"] + weather_test[weather_cols] @ weather_model.params[weather_cols])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, year in zip(axes, [2024, 2025]):
    df = weather_test[weather_test["Year"] == year].sort_values("Week")
    ax.plot(df["Week"], df["weather_multiplier"], marker="o", linewidth=2)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set(title=str(year), xlabel="Week of year", xticks=df["Week"])
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Weather multiplier")
fig.suptitle("Weekly weather effect, 2024–2025")
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
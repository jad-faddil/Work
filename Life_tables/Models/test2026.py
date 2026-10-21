import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from patsy import dmatrix

population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_all.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")
weather = pd.read_parquet("Life_tables/Life_tables_data/weekly_weather.parquet")

deaths = deaths.groupby(["Year", "Week", "Age"], as_index=False)["Deaths"].sum()
deaths = deaths[~((deaths["Year"] == 2026) & (deaths["Week"] == 51))].copy()
weather = weather.rename(columns={"year": "Year", "week": "Week"})
weather_cols = ["TN", "TM", "TX", "UN", "UM", "UX"]
weather = weather.groupby(["Year", "Week"])[weather_cols].mean().reset_index()

data = population.merge(deaths, on=["Year", "Week", "Age"], how="left")
data["Deaths"] = data["Deaths"].fillna(0).astype(int)
data["Population"] = data["Total"].round().astype(int)
data = data[data["Age"].between(20, 85) & data["Week"].between(18, 44)].merge(weather, on=["Year", "Week"], how="left")

# Baseline mortality rate by age, estimated through 2025
qbase = data[data["Year"].between(2019, 2024)].groupby("Age").apply(lambda g: g["Deaths"].sum() / g["Population"].sum(), include_groups=False)

# Fit seasonal model through 2025
train = data[data["Year"].between(2014, 2024) & data["Week"].between(18, 40)].copy()
train["qbase"] = train["Age"].map(qbase)

seasonal_model = smf.glm("Deaths ~ bs(Week, df=6, degree=3, include_intercept=False, lower_bound=18, upper_bound=40)", data=train, family=sm.families.Poisson(), offset=np.log(train["Population"] * train["qbase"])).fit()

# Predict seasonal multipliers for weeks 22–35
weeks = pd.DataFrame({"Week": np.arange(22, 36)})
X_weeks = dmatrix("bs(Week, df=6, degree=3, include_intercept=False, lower_bound=22, upper_bound=36)", weeks, return_type="dataframe")
seasonal_multiplier = pd.Series(np.exp(np.asarray(X_weeks) @ seasonal_model.params), index=weeks["Week"])
seasonal_multiplier /= seasonal_multiplier.mean()

# Evaluate May–September 2026
test = data[(data["Year"] == 2026) & data["Week"].between(22, 35) & data["Age"].between(20, 85)].copy()
test["qx_baseline"] = test["Age"].map(qbase)
test["qx_seasonal"] = test["qx_baseline"] * test["Week"].map(seasonal_multiplier)
test["Expected_baseline"] = test["Population"] * test["qx_baseline"]
test["Expected_seasonal"] = test["Population"] * test["qx_seasonal"]

# Overall results
for name, col in [("Baseline", "Expected_baseline"), ("Seasonal", "Expected_seasonal")]:
    expected = test[col].sum()
    observed = test["Deaths"].sum()
    print(f"{name}: expected={expected:.0f}, observed={observed:.0f}, excess={observed-expected:+.0f} ({100*(observed/expected-1):+.2f}%)")

print(f"\nObserved deaths: {test['Deaths'].sum():,.0f}")
print(f"Baseline excess: {test['Deaths'].sum()-test['Expected_baseline'].sum():+,.0f}")
print(f"Seasonal excess: {test['Deaths'].sum()-test['Expected_seasonal'].sum():+,.0f}")

# Breakdown by age
age_results = test.groupby("Age").agg(Observed=("Deaths", "sum"), Expected_baseline=("Expected_baseline", "sum"), Expected_seasonal=("Expected_seasonal", "sum")).reset_index()
age_results["Excess_baseline"] = age_results["Observed"] - age_results["Expected_baseline"]
age_results["Excess_seasonal"] = age_results["Observed"] - age_results["Expected_seasonal"]

# Weekly mortality residuals for 2026, weeks 22–35
weekly = test[test["Year"].eq(2026) & test["Week"].between(22, 35)].groupby("Week").agg(Observed=("Deaths", "sum"), Expected=("Expected_seasonal", "sum")).reset_index()
weekly["Difference"] = weekly["Observed"] - weekly["Expected"]
weekly["Difference_pct"] = 100 * weekly["Difference"] / weekly["Expected"]

# National weekly weather averages
weather_2026 = weather[weather["Year"].eq(2026) & weather["Week"].between(22, 35)].groupby("Week").agg(TN=("TN", "mean"), TX=("TX", "mean"), TM=("TM", "mean")).reset_index()

weekly = weekly.merge(weather_2026, on="Week", how="left")
print(weekly.sort_values("Difference", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.1f}"))

test_2026 = data[data["Year"].eq(2026) & data["Week"].between(22, 35) & data["Age"].between(20, 85)].copy()
test_2026["Expected_seasonal"] = test_2026["Population"] * test_2026["Age"].map(qbase) * test_2026["Week"].map(seasonal_multiplier)

weekly_diff = test_2026.groupby("Week").agg(Observed=("Deaths", "sum"), Expected=("Expected_seasonal", "sum"))
weekly_diff["Difference"] = weekly_diff["Observed"] - weekly_diff["Expected"]
selected_weeks = weekly_diff.nlargest(3, "Difference").index

ages = [25, 40, 55, 70, 80]

age_distribution = test_2026[test_2026["Week"].isin(selected_weeks) & test_2026["Age"].isin(ages)].groupby(["Week", "Age"]).agg(Observed=("Deaths", "sum"), Expected=("Expected_seasonal", "sum")).reset_index()

age_distribution["Difference"] = age_distribution["Observed"] - age_distribution["Expected"]
age_distribution["Difference_pct"] = 100 * age_distribution["Difference"] / age_distribution["Expected"]

print("Selected weeks:", sorted(selected_weeks))
print(age_distribution.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
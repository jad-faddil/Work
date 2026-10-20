import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

# Load data
population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_all.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")

# Aggregate deaths by year, week and age
deaths = deaths.groupby(["Year", "Week", "Age"], as_index=False)["Deaths"].sum()

# Build dataset
data = population.merge(deaths, on=["Year", "Week", "Age"], how="left")
data["Deaths"] = data["Deaths"].fillna(0).astype(int)
data["Population"] = data["Total"].round().astype(int)
data = data[data["Age"].between(20, 85) & data["Week"].between(22, 36)].copy()

# Baseline mortality rate by age, estimated over 2019–2024
base = data[data["Year"].between(2019, 2024)].groupby("Age").agg(Deaths=("Deaths", "sum"), Population=("Population", "sum"))
qbase = base["Deaths"] / base["Population"]

# Fit seasonal GLM over 2014–2024
train = data[data["Year"].between(2019, 2024)].copy()
train["qbase"] = train["Age"].map(qbase)
train["Week"] = train["Week"].astype("int64")

seasonal_model = smf.glm("Deaths ~ 0 + C(Week)", data=train, family=sm.families.Poisson(), offset=np.log(train["Population"] * train["qbase"])).fit()

# Extract one multiplier for each week, 22–36 inclusive
weeks = pd.DataFrame({"Week": np.arange(22, 37)})
X_weeks = dmatrix("0 + C(Week)", weeks, return_type="dataframe")
X_weeks = X_weeks.reindex(columns=seasonal_model.params.index, fill_value=0)

seasonal_multiplier = pd.Series(np.exp(X_weeks.to_numpy() @ seasonal_model.params.to_numpy()), index=weeks["Week"], name="Seasonal_multiplier")

print("Weekly seasonal multipliers:")
print(seasonal_multiplier)

# Evaluate 2025 and 2026
for year in [2025, 2026]:
    test = data[data["Year"] == year].copy()
    test["qx_baseline"] = test["Age"].map(qbase)
    test["Seasonal_multiplier"] = test["Week"].map(seasonal_multiplier)

    test["Expected_baseline"] = test["Population"] * test["qx_baseline"]
    test["Expected_seasonal"] = test["Expected_baseline"] * test["Seasonal_multiplier"]

    observed = test["Deaths"].sum()
    print(f"\n{year} — observed deaths: {observed:,.0f}")

    for name, col in [("Baseline", "Expected_baseline"), ("Seasonal", "Expected_seasonal")]:
        expected = test[col].sum()
        print(
            f"{name}: expected={expected:,.0f}, "
            f"excess={observed - expected:+,.0f} "
            f"({100 * (observed / expected - 1):+.2f}%)"
        )
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

# Load data
population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_all.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")
weather = pd.read_parquet("Life_tables/Life_tables_data/weekly_weather.parquet")

# Aggregate deaths and weather by year, week and age / week
deaths = deaths.groupby(["Year", "Week", "Age"], as_index=False)["Deaths"].sum()

training_years = range(2019, 2025)
training_weeks = range(22, 37)
weather_cols = ["TN", "TX", "UN", "UX"]
weather = weather.rename(columns={"year": "Year", "week": "Week"})
weather = weather.groupby(["Year", "Week"], as_index=False)[weather_cols].mean()

# Build dataset
data = population.merge(deaths, on=["Year", "Week", "Age"], how="left")
data["Deaths"] = data["Deaths"].fillna(0).astype(int)
data["Population"] = data["Total"].round().astype(int)
data = data[data["Age"].between(20, 85) & data["Week"].isin(training_weeks)].copy()
data = data.merge(weather, on=["Year", "Week"], how="left")

# Baseline mortality rate by age, estimated over 2019–2024
base = data[data["Year"].isin(training_years)].groupby("Age").agg(Deaths=("Deaths", "sum"), Population=("Population", "sum"))
qbase = base["Deaths"] / base["Population"]

# Center weather variables on the training-period means so their multiplicative
# contribution is interpreted relative to average training-period weather.
weather_means = data.loc[data["Year"].isin(training_years), weather_cols].mean()
for col in weather_cols:
    data[f"{col}_centered"] = data[col] - weather_means[col]
weather_terms = [f"{col}_centered" for col in weather_cols]

# Fit the original seasonal model
train = data[data["Year"].isin(training_years)].copy()
train["qbase"] = train["Age"].map(qbase)
train["Week"] = train["Week"].astype("int64")

seasonal_model = smf.glm("Deaths ~ 0 + C(Week)", data=train, family=sm.families.Poisson(), offset=np.log(train["Population"] * train["qbase"])).fit()

# Extract one seasonal multiplier for each week, 22–36 inclusive
weeks = pd.DataFrame({"Week": np.arange(22, 37)})
X_weeks = dmatrix("0 + C(Week)", weeks, return_type="dataframe")
X_weeks = X_weeks.reindex(columns=seasonal_model.params.index, fill_value=0)
seasonal_multiplier = pd.Series(np.exp(X_weeks.to_numpy() @ seasonal_model.params.to_numpy()), index=weeks["Week"], name="Seasonal_multiplier")

# Fit a joint seasonal + weather model. The week effects and weather effects
# are estimated together, with the same baseline offset.
weather_formula = "Deaths ~ 0 + C(Week) + " + " + ".join(weather_terms)
weather_train = train.dropna(subset=weather_terms).copy()
weather_model = smf.glm(weather_formula, data=weather_train, family=sm.families.Poisson(), offset=np.log(weather_train["Population"] * weather_train["qbase"])).fit()

# Evaluate 2025 and 2026
for year in [2025, 2026]:
    test = data[data["Year"] == year].copy()
    test["qx_baseline"] = test["Age"].map(qbase)
    test["Seasonal_multiplier"] = test["Week"].map(seasonal_multiplier)

    test["Expected_baseline"] = test["Population"] * test["qx_baseline"]
    test["Expected_seasonal"] = test["Expected_baseline"] * test["Seasonal_multiplier"]

    # Prediction from joint seasonal + weather model
    test["Expected_weather"] = np.nan
    valid_weather = test[weather_terms].notna().all(axis=1)
    test.loc[valid_weather, "Expected_weather"] = weather_model.predict(test.loc[valid_weather], offset=np.log(test.loc[valid_weather, "Population"] * test.loc[valid_weather, "qx_baseline"]))

    # Isolate the weather multiplier from the joint model by setting centered
    # weather variables to zero while keeping the fitted week effects fixed.
    weather_reference = test.loc[valid_weather].copy()
    for term in weather_terms:
        weather_reference[term] = 0.0
    reference_expected = weather_model.predict(weather_reference, offset=np.log(weather_reference["Population"] * weather_reference["qx_baseline"]))
    test.loc[valid_weather, "Weather_multiplier"] = (test.loc[valid_weather, "Expected_weather"].to_numpy() / np.asarray(reference_expected))

    observed = test["Deaths"].sum()
    print(f"\n{year} — observed deaths: {observed:,.0f}")

    for name, col in [("Baseline", "Expected_baseline"), ("Seasonal", "Expected_seasonal"), ("Seasonal + weather", "Expected_weather")]:
        expected = test[col].sum()
        print(
            f"{name}: expected={expected:,.0f}, "
            f"excess={observed - expected:+,.0f} "
            f"({100 * (observed / expected - 1):+.2f}%)"
        )

    if valid_weather.any():
        print(
            "Weather multiplier (min / mean / max): "
            f"{test.loc[valid_weather, 'Weather_multiplier'].min():.3f} / "
            f"{test.loc[valid_weather, 'Weather_multiplier'].mean():.3f} / "
            f"{test.loc[valid_weather, 'Weather_multiplier'].max():.3f}"
        )

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

# Load national weekly data
population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_all.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")
weather = pd.read_parquet("Life_tables/Life_tables_data/weekly_weather.parquet")

training_years = range(2019, 2025)
training_weeks = range(22, 37)  # Weeks 22–36 inclusive
weather_cols = ["TN", "TX", "UN", "UX"]

# Aggregate deaths by year, week and age; weather is national by year and week
deaths = deaths.groupby(["Year", "Week", "Age"], as_index=False)["Deaths"].sum()
weather = weather.rename(columns={"year": "Year", "week": "Week"})
weather = weather.groupby(["Year", "Week"], as_index=False)[weather_cols].mean()

data = population.merge(deaths, on=["Year", "Week", "Age"], how="left")
data["Deaths"] = data["Deaths"].fillna(0).astype(int)
data["Population"] = data["Total"].round().astype(int)
data = data[data["Age"].between(20, 85) & data["Week"].isin(training_weeks)].copy()
data = data.merge(weather, on=["Year", "Week"], how="left")
data["Year_centered"] = data["Year"] - np.mean(list(training_years))

train = data[data["Year"].isin(training_years)].copy()

base = train.groupby("Age").agg(Deaths=("Deaths", "sum"), Population=("Population", "sum"))
qbase = base["Deaths"] / base["Population"]
train["qbase"] = train["Age"].map(qbase)
train["Week"] = train["Week"].astype("int64")
data["Week"] = data["Week"].astype("int64")

seasonal_model = smf.glm("Deaths ~ 0 + C(Week)", data=train, family=sm.families.Poisson(), offset=np.log(train["Population"] * train["qbase"])).fit()

weeks = pd.DataFrame({"Week": np.arange(22, 37)})
X_weeks = dmatrix("0 + C(Week)", weeks, return_type="dataframe")
X_weeks = X_weeks.reindex(columns=seasonal_model.params.index, fill_value=0)
seasonal_multiplier = pd.Series(np.exp(X_weeks.to_numpy() @ seasonal_model.params.to_numpy()), index=weeks["Week"], name="Seasonal_multiplier")

train_weather = train[["Year", "Week", *weather_cols]].drop_duplicates(["Year", "Week"])
weather_mean = train_weather.groupby("Week")[weather_cols].mean()

for col in weather_cols:
    anomaly_col = f"{col}_anom"
    z_col = f"{col}_anom_z"
    data[anomaly_col] = data[col] - data["Week"].map(weather_mean[col])
    train_anomalies = train_weather[col] - train_weather["Week"].map(weather_mean[col])
    anomaly_sd = train_anomalies.std(ddof=0)
    data[z_col] = data[anomaly_col] / anomaly_sd

weather_terms = [f"{col}_anom_z" for col in weather_cols]
joint_formula = "Deaths ~ C(Age) + C(Week) + Year_centered + " + " + ".join(weather_terms)
joint_train = data[data["Year"].isin(training_years)].dropna(subset=weather_terms).copy()

joint_model = smf.glm(joint_formula, data=joint_train, family=sm.families.Poisson(), offset=np.log(joint_train["Population"])).fit(maxiter=100)

print("Joint model fitted on years 2019–2024, weeks 22–36.")
print(f"Training rows: {len(joint_train):,}")
print("Weather coefficients (log-rate change per 1 SD weekly anomaly):")
print(joint_model.params[weather_terms].to_string(float_format=lambda x: f"{x:+.4f}"))
print(f"Estimated annual trend multiplier: {np.exp(joint_model.params['Year_centered']):.4f}")

for year in [2025, 2026]:
    test_all = data[data["Year"] == year].copy()
    test = test_all.dropna(subset=weather_terms).copy()
    if len(test) != len(test_all):
        print(f"\n{year}: excluding {len(test_all) - len(test):,} rows with missing weather from all model comparisons.")

    test["qx_baseline"] = test["Age"].map(qbase)
    test["Seasonal_multiplier"] = test["Week"].map(seasonal_multiplier)
    test["Expected_baseline"] = test["Population"] * test["qx_baseline"]
    test["Expected_seasonal"] = test["Expected_baseline"] * test["Seasonal_multiplier"]

    # Prediction with observed weather anomalies
    test["Expected_joint_weather"] = joint_model.predict(test, offset=np.log(test["Population"]))

    # Counterfactual prediction at normal (training-period average) weather
    reference = test.copy()
    reference[weather_terms] = 0.0
    test["Expected_joint_reference"] = joint_model.predict(reference, offset=np.log(reference["Population"]))
    test["Weather_multiplier_joint"] = (test["Expected_joint_weather"] / test["Expected_joint_reference"])

    observed = test["Deaths"].sum()
    print(f"\n{year} — observed deaths: {observed:,.0f}")
    for name, col in [("Joint model, normal weather", "Expected_joint_reference"), ("Joint model, observed weather", "Expected_joint_weather")]:
        expected = test[col].sum()
        print(f"{name}: expected={expected:,.0f}, "
            f"observed - expected={observed - expected:+,.0f} "
        )

    multipliers = test.groupby("Week")["Weather_multiplier_joint"].first()
    print("Joint weather multiplier by week (min / mean / max): "f"{multipliers.min():.3f} / {multipliers.mean():.3f} / {multipliers.max():.3f}")

    weekly = test.groupby("Week").agg(
        Observed=("Deaths", "sum"), 
        Joint_normal_weather=("Expected_joint_reference", "sum"), 
        Joint_observed_weather=("Expected_joint_weather", "sum"),
        Weather_multiplier=("Weather_multiplier_joint", "first"),
        ).reset_index()
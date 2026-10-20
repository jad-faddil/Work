import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_departements.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")
weather = pd.read_parquet("Life_tables/Life_tables_data/weekly_weather.parquet")

# Weather predictors used by the joint weather model
weather_cols = ["TN", "TX", "TM", "UN", "UX", "UM"]
weather_keys = ["Departement", "Year", "Week"]
training_years = range(2019, 2025)

# Aggregate deaths across any remaining dimensions, e.g. sex
keys = ["Year", "Week", "Age", "Departement"]
deaths = deaths.groupby(keys, as_index=False)["Deaths"].sum()

# Build departmental dataset
population = population[keys + ["Total"]].rename(columns={"Total": "Population"})
data = population.merge(deaths, on=keys, how="left")
data["Deaths"] = data["Deaths"].fillna(0).astype(int)
data["Population"] = data["Population"].round().astype(int)

# Merge weekly weather onto every age group in the matching department-week
weather = weather[weather_keys + weather_cols]
data = data.merge(weather, on=weather_keys, how="left", validate="many_to_one")

# Restrict to the same ages and weeks as the baseline / seasonal models
data = data[data["Age"].between(20, 85) & data["Week"].between(22, 36)].copy()

# Baseline mortality rate by department and age, estimated over 2019–2024
base = data[data["Year"].isin(training_years)].groupby(["Departement", "Age"], as_index=False).agg(Deaths=("Deaths", "sum"), Population=("Population", "sum"))
base["qbase"] = base["Deaths"] / base["Population"]

data = data.merge(base[["Departement", "Age", "qbase"]], on=["Departement", "Age"], how="left", validate="many_to_one")
print("Department-age cells with zero baseline mortality:", (base["qbase"] == 0).sum())

# Fit one seasonal and one joint seasonal + weather model per department
all_predictions = []
all_multipliers = []
results = []

for dep, dept_data in data.groupby("Departement", sort=True):
    train = dept_data[dept_data["Year"].isin(training_years)].copy()
    train = train[(train["qbase"] > 0) & (train["Population"] > 0)].copy()
    train["Week"] = train["Week"].astype("int64")

    offset_train = np.log(train["Population"] * train["qbase"])
    seasonal_model = smf.glm("Deaths ~ 0 + C(Week)", data=train, family=sm.families.Poisson(), offset=offset_train).fit()
    seasonal_multiplier = pd.Series({int(term.rsplit("[", 1)[1].rstrip("]")): np.exp(coef) for term, coef in seasonal_model.params.items()},name="Seasonal_multiplier")

    mult = seasonal_multiplier.rename_axis("Week").reset_index()
    mult["Departement"] = dep
    all_multipliers.append(mult[["Departement", "Week", "Seasonal_multiplier"]])

    weather_train = train.copy()
    weather_formula = "Deaths ~ 0 + C(Week) + " + " + ".join(weather_cols)
    weather_model = smf.glm(weather_formula, data=weather_train, family=sm.families.Poisson(), offset=np.log(weather_train["Population"] * weather_train["qbase"])).fit()

    # Evaluate 2025 and 2026
    for year in [2025, 2026]:
        test = dept_data[dept_data["Year"] == year].copy()
        test["Week"] = test["Week"].astype("int64")
        test["Seasonal_multiplier"] = test["Week"].map(seasonal_multiplier)

        test["Expected_baseline"] = test["Population"] * test["qbase"]
        test["Expected_seasonal"] = test["Expected_baseline"] * test["Seasonal_multiplier"]
        test["Expected_weather"] = 0.0
        valid = test["qbase"].gt(0) & test["Population"].gt(0)
        test.loc[valid, "Expected_weather"] = weather_model.predict(test.loc[valid], offset=np.log(test.loc[valid, "Population"] * test.loc[valid, "qbase"]))

        observed = int(test["Deaths"].sum())
        expected_base = test["Expected_baseline"].sum()
        expected_seasonal = test["Expected_seasonal"].sum()
        expected_weather = test["Expected_weather"].sum()
        expected_base_int = int(expected_base)
        expected_seasonal_int = int(expected_seasonal)
        expected_weather_int = int(expected_weather)

        results.append({
            "Departement": dep,
            "Year": year,
            "Observed": observed,
            "Expected_baseline": expected_base_int,
            "Excess_baseline": observed - expected_base_int,
            "Excess_baseline_pct": 100 * (observed / expected_base_int - 1) if expected_base_int > 0 else np.nan,
            "Expected_seasonal": expected_seasonal_int,
            "Excess_seasonal": observed - expected_seasonal_int,
            "Excess_seasonal_pct": 100 * (observed / expected_seasonal_int - 1) if expected_seasonal_int > 0 else np.nan,
            "Expected_weather": expected_weather_int,
            "Excess_weather": observed - expected_weather_int,
            "Excess_weather_pct": 100 * (observed / expected_weather_int - 1) if expected_weather_int > 0 else np.nan
        })

        all_predictions.append(test[["Year", "Deaths", "Expected_baseline", "Expected_seasonal", "Expected_weather"]].copy())
        
# Save results
multipliers = pd.concat(all_multipliers, ignore_index=True)
summary = pd.DataFrame(results)

predictions = pd.concat(all_predictions, ignore_index=True)

columns = ["Observed", "Expected_baseline", "Expected_seasonal", "Expected_weather", "Excess_baseline", "Excess_seasonal", "Excess_weather"]

print("\nOverall results by year:")
print(summary.groupby("Year")[columns].sum().round(1))

for year, df in predictions.groupby("Year"):
    print(f"\n{year}")
    for model in ["baseline", "seasonal", "weather"]:
        error = df["Deaths"] - df[f"Expected_{model}"]
        mae = error.abs().mean()
        rmse = np.sqrt((error ** 2).mean())
        print(f"{model:10s} MAE = {mae:.3f} | RMSE = {rmse:.3f}")

print(multipliers.describe(percentiles=[.05, .25, .5, .75, .95]))
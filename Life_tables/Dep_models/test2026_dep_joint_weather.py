import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

training_years = range(2019, 2025)
training_weeks = range(22, 37)
training_ages = range(20, 86)
test_years = [2025, 2026]
weather_cols = ["TN", "TX", "UN", "UX"]
anom_cols = [f"{c}_anom_z" for c in weather_cols]
year_center = np.mean(list(training_years))
model_years = [*training_years, *test_years]

population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_departements.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")
weather = pd.read_parquet("Life_tables/Life_tables_data/weekly_weather.parquet")

for df in (population, deaths, weather):
    df["Departement"] = df["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

population = population[population["Year"].isin(model_years) & population["Week"].isin(training_weeks) & population["Age"].isin(training_ages)].copy()
deaths = deaths[deaths["Year"].isin(model_years) & deaths["Week"].isin(training_weeks) & deaths["Age"].isin(training_ages)]
deaths = deaths.groupby(["Year", "Week", "Age", "Departement"], as_index=False)["Deaths"].sum()
weather = weather[weather["Year"].isin(model_years) & weather["Week"].isin(training_weeks)]
weather = weather.groupby(["Departement", "Year", "Week"], as_index=False)[weather_cols].mean()

data = population.merge(deaths, on=["Year", "Week", "Age", "Departement"], how="left")
data = data.merge(weather, on=["Departement", "Year", "Week"], how="left")
data["Deaths"] = data["Deaths"].fillna(0)
data["Population"] = data["Total"].round().astype(int)
data = data[data["Population"] > 0].copy()

formula = (f"Deaths ~ 0 + C(Age) + C(Week) + I(Year - {year_center:g}) + " + " + ".join(anom_cols))

summary, weekly_results, age_results, coefficients = [], [], [], []

for dep, d in data.groupby("Departement"):
    train = d[d["Year"].isin(training_years)].copy()
    test = d[d["Year"].isin(test_years)].copy()

    # Standardize each weather variable against the same week in 2019–2024
    weather_ref = train[["Year", "Week", *weather_cols]].drop_duplicates(["Year", "Week"])
    means = weather_ref.groupby("Week")[weather_cols].mean()
    stds = weather_ref.groupby("Week")[weather_cols].std().replace(0, np.nan)

    for c in weather_cols:
        train[f"{c}_anom_z"] = (train[c] - train["Week"].map(means[c])) / train["Week"].map(stds[c])
        test[f"{c}_anom_z"] = (test[c] - test["Week"].map(means[c])) / test["Week"].map(stds[c])

    model = smf.glm(formula, data=train, family=sm.families.Poisson(), offset=np.log(train["Population"])).fit()

    normal_test = test.copy()
    normal_test[anom_cols] = 0
    test["Expected_normal"] = model.predict(normal_test, offset=np.log(normal_test["Population"]))
    test["Expected_weather"] = model.predict(test, offset=np.log(test["Population"]))
    age_results.append(test[["Departement", "Year", "Week", "Age", "Population", "Deaths", "Expected_weather"]].copy())

    trend_coef = model.params[f"I(Year - {year_center:g})"]
    coefficients.append({"Departement": dep, "Term": "Annual_trend", "Coefficient": trend_coef, "Multiplier": np.exp(trend_coef)})
    coefficients.extend({"Departement": dep, "Term": c, "Coefficient": model.params[c], "Multiplier": np.exp(model.params[c])} for c in anom_cols)

    for year, t in test.groupby("Year"):
        obs, normal, actual = t["Deaths"].sum(), t["Expected_normal"].sum(), t["Expected_weather"].sum()
        summary.append({
            "Departement": dep, "Year": year, "Observed": obs,
            "Joint_normal_weather": normal, "Joint_observed_weather": actual,
            "Bias_normal_pct": 100 * (obs / normal - 1),
            "Bias_weather_pct": 100 * (obs / actual - 1),
            "Weather_multiplier": actual / normal,
            "Annual_trend_multiplier": np.exp(trend_coef)
        })

    weekly = test.groupby(["Year", "Week"], as_index=False).agg(Observed=("Deaths", "sum"), Joint_normal_weather=("Expected_normal", "sum"), Joint_observed_weather=("Expected_weather", "sum"))
    weekly["Departement"] = dep
    weekly["Weather_multiplier"] = weekly["Joint_observed_weather"] / weekly["Joint_normal_weather"]
    weekly_results.append(weekly)

age_results = pd.concat(age_results, ignore_index=True)
age_results.to_parquet("Life_tables/Life_tables_data/dep_glm_age_predictions.parquet", index=False)

weekly_results = pd.concat(weekly_results, ignore_index=True)
weekly_results.to_parquet("Life_tables/Life_tables_data/dep_glm_predictions.parquet", index=False)

print("Saved files")
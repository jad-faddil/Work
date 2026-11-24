import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

training_years = range(1994, 2004)
training_weeks = range(22, 37)
training_ages = range(20, 86)
test_years = range(2026, 2027)
weather_cols = ["TX", "TN"]
diff_cols = [f"{c}_diff" for c in weather_cols]
year_center = np.mean(list(training_years))
model_years = [*training_years, *test_years]

REG_TO_DEPS = {
    "11": ["75", "77", "78", "91", "92", "93", "94", "95"],  # Île-de-France
    "24": ["18", "28", "36", "37", "41", "45"],  # Centre-Val de Loire
    "27": ["21", "25", "39", "58", "70", "71", "89", "90"],  # Bourgogne-Franche-Comté
    "28": ["14", "27", "50", "61", "76"],  # Normandie
    "32": ["02", "59", "60", "62", "80"],  # Hauts-de-France
    "44": ["08", "10", "51", "52", "54", "55", "57", "67", "68", "88"],  # Grand Est
    "52": ["44", "49", "53", "72", "85"],  # Pays de la Loire
    "53": ["22", "29", "35", "56"],  # Bretagne
    "75": ["16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87"],  # Nouvelle-Aquitaine
    "76": ["09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82"],  # Occitanie
    "84": ["01", "03", "07", "15", "26", "38", "42", "43", "63", "69", "73", "74"],  # Auvergne-Rhône-Alpes
    "93": ["04", "05", "06", "13", "83", "84"],  # Provence-Alpes-Côte d'Azur
    "94": ["20"]  # Corse
}

DEP_TO_REG = {dep: reg for reg, deps in REG_TO_DEPS.items() for dep in deps}

population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_departements.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")
weather = pd.read_parquet("Life_tables/Life_tables_data/weekly_weather.parquet")
ira = pd.read_csv("Life_tables/Life_tables_data/infections.csv")

ira["geo_insee"] = ira["geo_insee"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
ira = ira.sort_values(["geo_insee", "Year", "Week"])
ira["inc100_lag1"] = ira.groupby("geo_insee")["inc100"].shift(1)

train_ira_mask = ira["Year"].isin(training_years) & ira["Week"].isin(training_weeks)
ira_floor = (ira.loc[train_ira_mask].groupby("geo_insee")["inc100_lag1"].quantile(0.20))
ira["IRA_excess_lag1"] = (ira["inc100_lag1"] - ira["geo_insee"].map(ira_floor)).clip(lower=0)

for df in (population, deaths, weather):
    df["Departement"] = df["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

population = population[population["Year"].isin(model_years) & population["Week"].isin(training_weeks) & population["Age"].isin(training_ages)].copy()
deaths = deaths[deaths["Year"].isin(model_years) & deaths["Week"].isin(training_weeks) & deaths["Age"].isin(training_ages)]
deaths = deaths.groupby(["Year", "Week", "Age", "Departement"], as_index=False)["Deaths"].sum()
weather = weather[weather["Year"].isin(model_years) & weather["Week"].isin(training_weeks)]
weather = weather.groupby(["Departement", "Year", "Week"], as_index=False)[weather_cols].mean()
weather = weather.sort_values(["Departement", "Year", "Week"]).reset_index(drop=True)

train_w_mask = weather["Year"].isin(training_years) & weather["Week"].isin(training_weeks)
tx_thresh = weather.loc[train_w_mask].groupby("Departement")["TX"].quantile(0.90)
tn_thresh = weather.loc[train_w_mask].groupby("Departement")["TN"].quantile(0.90)

weather["TX_excess"] = (weather["TX"] - weather["Departement"].map(tx_thresh)).clip(lower=0)
weather["TN_excess"] = (weather["TN"] - weather["Departement"].map(tn_thresh)).clip(lower=0)

weather["TX_excess_lag1"] = weather.groupby("Departement")["TX_excess"].shift(1).fillna(0.0)
weather["TN_excess_lag1"] = weather.groupby("Departement")["TN_excess"].shift(1).fillna(0.0)

data = population.merge(deaths, on=["Year", "Week", "Age", "Departement"], how="left")
data = data.merge(weather, on=["Departement", "Year", "Week"], how="left")
data["geo_insee"] = data["Departement"].map(DEP_TO_REG)
data = data.merge(ira[["geo_insee", "Year", "Week", "IRA_excess_lag1"]], on=["geo_insee", "Year", "Week"], how="left", validate="many_to_one")

data["Deaths"] = data["Deaths"].fillna(0)
data["Population"] = data["Total"].round().astype(int)
data = data[data["Population"] > 0].copy()

age_bins = [*range(20, 65, 5), *range(65, 87, 3)]
age_labels = [f"{a}-{b - 1}" for a, b in zip(age_bins[:-1], age_bins[1:])]

data["Age_group"] = pd.cut(data["Age"], bins=age_bins, labels=age_labels, right=False,)
passthrough_cols = ["geo_insee", "IRA_excess_lag1", "TX_excess", "TN_excess", "TX_excess_lag1", "TN_excess_lag1", *weather_cols, *(c for c in diff_cols if c in data.columns)]
data = (data.groupby(["Departement", "Year", "Week", "Age_group"], as_index=False, observed=True).agg(
    Deaths=("Deaths", "sum"), Population=("Population", "sum"), **{col: (col, "first") for col in passthrough_cols}))

data["IRA_excess_lag1"] = np.where(data["Year"] < 2020, 0.0, data["IRA_excess_lag1"])

formula = f"Deaths ~ 0 + C(Age_group) + C(Week) + I(Year - {year_center:g}) + TX_diff:Age_group + TX_excess + TN_excess + TX_excess_lag1 + TN_excess_lag1 + " + " + ".join(diff_cols)

summary, weekly_results, age_results, coefficients = [], [], [], []

for dep, d in data.groupby("Departement"):
    train = d[d["Year"].isin(training_years)].copy()
    test = d[d["Year"].isin(test_years)].copy()

    # Standardize each weather variable against the same week in 2019–2024
    weather_ref = train[["Year", "Week", *weather_cols]].drop_duplicates(["Year", "Week"])
    means = weather_ref.groupby("Week")[weather_cols].mean()
    stds = weather_ref.groupby("Week")[weather_cols].std().replace(0, np.nan)

    for c in weather_cols:
        train[f"{c}_diff"] = (train[c] - train["Week"].map(means[c])) / train["Week"].map(stds[c])
        test[f"{c}_diff"] = (test[c] - test["Week"].map(means[c])) / test["Week"].map(stds[c])

    model = smf.glm(formula, data=train, family=sm.families.Poisson(), offset=np.log(train["Population"])).fit()

    normal_test = test.copy()
    normal_test[diff_cols] = 0
    test["Expected_normal"] = model.predict(normal_test, offset=np.log(normal_test["Population"]))
    test["Expected_weather"] = model.predict(test, offset=np.log(test["Population"]))
    age_results.append(test[["Departement", "Year", "Week", "Age_group", "Population", "Deaths", "Expected_normal", "Expected_weather"]].copy())

    trend_coef = model.params[f"I(Year - {year_center:g})"]
    coefficients.append({"Departement": dep, "Term": "Annual_trend", "Coefficient": trend_coef, "Multiplier": np.exp(trend_coef)})
    coefficients.extend({"Departement": dep, "Term": c, "Coefficient": model.params[c], "Multiplier": np.exp(model.params[c])} for c in diff_cols)

    for year, t in test.groupby("Year"):
        obs, normal, actual = t["Deaths"].sum(), t["Expected_normal"].sum(), t["Expected_weather"].sum()
        summary.append({
            "Departement": dep, "Year": year, "Observed": obs,
            "Normal_weather": normal, "Observed_weather": actual,
            "Bias_normal_pct": 100 * (obs / normal - 1),
            "Bias_weather_pct": 100 * (obs / actual - 1),
            "Weather_multiplier": actual / normal,
            "Annual_trend_multiplier": np.exp(trend_coef)
        })

    weekly = test.groupby(["Year", "Week"], as_index=False).agg(Observed=("Deaths", "sum"), Expected_deps_normal=("Expected_normal", "sum"), Expected_deps=("Expected_weather", "sum"))
    weekly["Departement"] = dep
    weekly["Weather_multiplier"] = weekly["Expected_deps"] / weekly["Expected_deps_normal"]
    weekly_results.append(weekly)

age_results = pd.concat(age_results, ignore_index=True)
age_results.to_parquet("Life_tables/Life_tables_data/dep_glm_age_predictions.parquet", index=False)

weekly_results = pd.concat(weekly_results, ignore_index=True)
weekly_results.to_parquet("Life_tables/Life_tables_data/dep_glm_predictions.parquet", index=False)

print("Saved files")
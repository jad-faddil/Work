import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

base = "Life_tables/Life_tables_data/"
training_years = range(2019, 2025)
test_years = [2025, 2026]
training_ages = range(20,87)
model_years = [*training_years, *test_years]
training_weeks = range(22, 37)

weather_cols = ["TN", "TM", "TX", "UN", "UM", "UX"]
anom_cols = [f"{c}_anom_z" for c in weather_cols]
year_center = np.mean(list(training_years))
eps = 1e-8

# 1. Load and clean raw data
pop_nat = pd.read_parquet(base + "weekly_population_all.parquet")
pop_dep = pd.read_parquet(base + "weekly_population_departements.parquet")
deaths = pd.read_parquet(base + "deaths_weekly.parquet")
weather = pd.read_parquet(base + "weekly_weather.parquet")

for df in (pop_dep, deaths, weather):
    df["Departement"] = df["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

# Filter to relevant years, weeks, and ages (20-86 needed to get qx at age 85)
pop_nat = pop_nat[pop_nat["Year"].isin(model_years) & pop_nat["Week"].isin(training_weeks) & pop_nat["Age"].isin(training_ages)].copy()
pop_dep = pop_dep[pop_dep["Year"].isin(model_years) & pop_dep["Week"].isin(training_weeks) & pop_dep["Age"].isin(training_ages)].copy()
deaths = deaths[deaths["Year"].isin(model_years) & deaths["Week"].isin(training_weeks) & deaths["Age"].isin(training_ages)]
deaths = deaths.groupby(["Year", "Week", "Age", "Departement"], as_index=False)["Deaths"].sum()
weather_nat = weather[weather["Year"].isin(model_years) & weather["Week"].isin(training_weeks)].groupby(["Year", "Week"], as_index=False)[weather_cols].mean()

# 2. Prepare national dataset & standardize weather using training years only
deaths_nat = deaths.groupby(["Year", "Week", "Age"], as_index=False)["Deaths"].sum()
nat = pop_nat[["Year", "Week", "Age", "Total"]].rename(columns={"Total": "Population"})
nat = nat.merge(deaths_nat, on=["Year", "Week", "Age"], how="left").merge(weather_nat, on=["Year", "Week"], how="left")
nat["Deaths"] = nat["Deaths"].fillna(0)
nat["Population"] = nat["Population"].round()

weather_train = weather_nat[weather_nat["Year"].isin(training_years)]
means = weather_train.groupby("Week")[weather_cols].mean()
stds = weather_train.groupby("Week")[weather_cols].std().replace(0, np.nan)

for c in weather_cols:
    nat[f"{c}_anom_z"] = (nat[c] - nat["Week"].map(means[c])) / nat["Week"].map(stds[c])

# 3. Fit National GLM on training years
nat_train = nat[nat["Year"].isin(training_years)].copy()
formula = f"Deaths ~ 0 + C(Age) + C(Week) + I(Year - {year_center:g}) + " + " + ".join(anom_cols)
national_model = smf.glm(formula, data=nat_train, family=sm.families.Poisson(), offset=np.log(nat_train["Population"])).fit()
national_model.save(base + "national_mortality_model.pkl")

# 4. Fit Brass coefficients on pooled training life tables
nat_life = nat_train.groupby("Age", as_index=False).agg(Population=("Population", "sum"), Deaths=("Deaths", "sum")).sort_values("Age")
nat_life["qx"] = (nat_life["Deaths"] / nat_life["Population"].replace(0, np.nan)).clip(0, 1)
nat_life["lx_nat"] = (1 - nat_life["qx"]).cumprod().shift(1, fill_value=1)

dep_train = pop_dep[pop_dep["Year"].isin(training_years)][["Year", "Week", "Age", "Departement", "Total"]].rename(columns={"Total": "Population"})
dep_train = dep_train.merge(deaths[deaths["Year"].isin(training_years)], on=["Year", "Week", "Age", "Departement"], how="left").fillna({"Deaths": 0})

dep_life = dep_train.groupby(["Departement", "Age"], as_index=False).agg(Population=("Population", "sum"), Deaths=("Deaths", "sum")).sort_values(["Departement", "Age"])
dep_life["qx"] = (dep_life["Deaths"] / dep_life["Population"].replace(0, np.nan)).clip(0, 1)
dep_life["lx_dep"] = dep_life.groupby("Departement")["qx"].transform(lambda q: (1 - q).cumprod().shift(1, fill_value=1))

brass = dep_life.merge(nat_life[["Age", "lx_nat"]], on="Age", how="inner")
brass = brass[brass["Age"].isin(training_ages[1:])].copy()
brass["x"] = np.log(brass["lx_nat"].clip(eps, 1 - eps) / (1 - brass["lx_nat"].clip(eps, 1 - eps)))
brass["y"] = np.log(brass["lx_dep"].clip(eps, 1 - eps) / (1 - brass["lx_dep"].clip(eps, 1 - eps)))

def fit_brass(g):
    x, y = g["x"].to_numpy(), g["y"].to_numpy()
    if len(g) < 3 or np.var(x) < 1e-12:
        return pd.Series({"alpha": np.nan, "beta": np.nan, "n_obs": len(g), "r_squared": np.nan})
    beta, alpha = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - alpha - beta * x) ** 2) / np.sum((y - y.mean()) ** 2)
    return pd.Series({"alpha": alpha, "beta": beta, "n_obs": len(g), "r_squared": r2})

brass_params = brass.groupby("Departement").apply(fit_brass, include_groups=False).reset_index()
brass_params.to_parquet(base + "brass_coefficients.parquet", index=False)

# 5. Predict on Test Years (2025-2026) and apply Brass transformation
nat_test = nat[nat["Year"].isin(test_years)].copy()
nat_test["Expected_national"] = national_model.predict(nat_test, offset=np.log(nat_test["Population"]))
nat_test["qx_nat"] = (nat_test["Expected_national"] / nat_test["Population"]).clip(0, 1)
nat_test = nat_test.sort_values(["Year", "Week", "Age"])
nat_test["lx_nat"] = nat_test.groupby(["Year", "Week"])["qx_nat"].transform(lambda q: (1 - q).cumprod().shift(1, fill_value=1))

dep_test = pop_dep[pop_dep["Year"].isin(test_years)][["Year", "Week", "Age", "Departement", "Total"]].rename(columns={"Total": "Population"})
dep_test = (
    dep_test.merge(nat_test[["Year", "Week", "Age", "lx_nat"]], on=["Year", "Week", "Age"], how="left")
    .merge(brass_params[["Departement", "alpha", "beta"]], on="Departement", how="left")
    .merge(deaths[deaths["Year"].isin(test_years)], on=["Year", "Week", "Age", "Departement"], how="left")
    .fillna({"Deaths": 0})
)

lx_clipped = dep_test["lx_nat"].clip(eps, 1 - eps)
dep_test["lx_dep"] = 1 / (1 + np.exp(-(dep_test["alpha"] + dep_test["beta"] * np.log(lx_clipped / (1 - lx_clipped)))))
dep_test.loc[dep_test["Age"] == 20, "lx_dep"] = 1.0
dep_test = dep_test.sort_values(["Departement", "Year", "Week", "Age"])
dep_test["lx_next"] = dep_test.groupby(["Departement", "Year", "Week"])["lx_dep"].shift(-1)
dep_test["qx_dep"] = 1 - dep_test["lx_next"] / dep_test["lx_dep"]
dep_test["Expected_brass"] = dep_test["Population"] * dep_test["qx_dep"]

# 6. Filter to evaluation ages (20-85) and save predictions
nat_predictions = nat_test[nat_test["Age"].isin(training_ages[:-1])].copy()
brass_predictions = dep_test[dep_test["Age"].isin(training_ages[:-1])].copy()

nat_predictions.to_parquet(base + "national_mortality_predictions.parquet", index=False)
brass_predictions.to_parquet(base + "brass_mortality_predictions.parquet", index=False)

print("National model converged:", national_model.converged)
print("Saved national_mortality_predictions.parquet and brass_mortality_predictions.parquet")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

base = "Life_tables/Life_tables_data/"
training_years, training_weeks = range(2019, 2025), range(22, 37)
weather_cols = ["TN", "TM", "TX", "UN", "UM", "UX"]
anom_cols = [f"{c}_anom_z" for c in weather_cols]
year_center = np.mean(list(training_years))
eps = 1e-8

pop_nat = pd.read_parquet(base + "weekly_population_all.parquet")
pop_dep = pd.read_parquet(base + "weekly_population_departements.parquet")
deaths = pd.read_parquet(base + "deaths_weekly.parquet")
weather = pd.read_parquet(base + "weekly_weather.parquet")

for df in (pop_dep, deaths, weather):
    df["Departement"] = df["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

pop_nat = pop_nat[pop_nat["Year"].isin(training_years) & pop_nat["Week"].isin(training_weeks) & pop_nat["Age"].between(20, 86)].copy()
pop_dep = pop_dep[pop_dep["Year"].isin(training_years) & pop_dep["Week"].isin(training_weeks) & pop_dep["Age"].between(20, 86)].copy()
deaths = deaths[deaths["Year"].isin(training_years) & deaths["Week"].isin(training_weeks) & deaths["Age"].between(20, 86)]
deaths = deaths.groupby(["Year", "Week", "Age", "Departement"], as_index=False)["Deaths"].sum()
weather = weather[weather["Year"].isin(training_years) & weather["Week"].isin(training_weeks)].groupby(["Year", "Week"], as_index=False)[weather_cols].mean()

# Fit national mortality model
deaths_nat = deaths.groupby(["Year", "Week", "Age"], as_index=False)["Deaths"].sum()
nat = pop_nat[["Year", "Week", "Age", "Total"]].rename(columns={"Total": "Population"})
nat = nat.merge(deaths_nat, on=["Year", "Week", "Age"], how="left").merge(weather, on=["Year", "Week"], how="left")
nat["Deaths"] = nat["Deaths"].fillna(0)
nat["Population"] = nat["Population"].round()

means = weather.groupby("Week")[weather_cols].mean()
stds = weather.groupby("Week")[weather_cols].std().replace(0, np.nan)
for c in weather_cols:
    nat[f"{c}_anom_z"] = (nat[c] - nat["Week"].map(means[c])) / nat["Week"].map(stds[c])

if nat[anom_cols].isna().any().any():
    raise ValueError("Missing or undefined weather anomalies.")

formula = f"Deaths ~ 0 + C(Age) + C(Week) + I(Year - {year_center:g}) + " + " + ".join(anom_cols)
national_model = smf.glm(formula, data=nat, family=sm.families.Poisson(), offset=np.log(nat["Population"])).fit()
national_model.save(base + "national_mortality_model.pkl")

# Build one pooled national life table from observed training data
nat_life = nat.groupby("Age", as_index=False).agg(Population=("Population", "sum"), Deaths=("Deaths", "sum"))
nat_life["qx"] = (nat_life["Deaths"] / nat_life["Population"].replace(0, np.nan)).clip(0, 1)
nat_life = nat_life.sort_values("Age")
nat_life["lx_nat"] = (1 - nat_life["qx"]).cumprod().shift(1, fill_value=1)

# Build one pooled observed life table per department
dep = pop_dep[["Year", "Week", "Age", "Departement", "Total"]].rename(columns={"Total": "Population"})
dep = dep.merge(deaths, on=["Year", "Week", "Age", "Departement"], how="left")
dep["Deaths"] = dep["Deaths"].fillna(0)

dep_life = dep.groupby(["Departement", "Age"], as_index=False).agg(Population=("Population", "sum"), Deaths=("Deaths", "sum"))
dep_life["qx"] = (dep_life["Deaths"] / dep_life["Population"].replace(0, np.nan)).clip(0, 1)
dep_life = dep_life.sort_values(["Departement", "Age"])
dep_life["lx_dep"] = dep_life.groupby("Departement")["qx"].transform(lambda q: (1 - q).cumprod().shift(1, fill_value=1))

# Fit Brass coefficients to the pooled curves
brass = dep_life.merge(nat_life[["Age", "lx_nat"]], on="Age", how="inner")
brass = brass[brass["Age"].between(21, 86)].copy()

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

print("National model converged:", national_model.converged)
print("Departments with Brass coefficients:", brass_params["beta"].notna().sum())
print(brass_params[["alpha", "beta", "n_obs", "r_squared"]].describe().round(4))
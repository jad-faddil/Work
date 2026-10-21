import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.iolib.smpickle import load_pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

training_years, test_years, training_weeks = range(2019, 2025), [2025, 2026], range(22, 37)
model_years = [*training_years, *test_years]
weather_cols = ["TN", "TM", "TX", "UN", "UM", "UX"]
anom_cols = [f"{c}_anom_z" for c in weather_cols]
base = "Life_tables/Life_tables_data/"

# Load fitted models and data
model = load_pickle(base + "national_mortality_model.pkl")
brass = pd.read_parquet(base + "brass_coefficients.parquet")
pop_nat = pd.read_parquet(base + "weekly_population_all.parquet")
pop_dep = pd.read_parquet(base + "weekly_population_departements.parquet")
deaths = pd.read_parquet(base + "deaths_weekly.parquet")
weather = pd.read_parquet(base + "weekly_weather.parquet")

for df in (pop_dep, deaths, weather):
    df["Departement"] = df["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

# Prepare test population and weather
nat = pop_nat.query("Year in @test_years and Week in @training_weeks and Age >= 20 and Age <= 86").copy()
dep = pop_dep.query("Year in @test_years and Week in @training_weeks and Age >= 20 and Age <= 86").copy()
weather = weather.query("Year in @model_years and Week in @training_weeks").copy()
weather_nat = weather.groupby(["Year", "Week"], as_index=False)[weather_cols].mean()

# Recreate training weather references exactly as in training
weather_train = weather_nat[weather_nat["Year"].isin(training_years)]
means = weather_train.groupby("Week")[weather_cols].mean()
stds = weather_train.groupby("Week")[weather_cols].std().replace(0, np.nan)

nat = nat.rename(columns={"Total": "Population"}).merge(weather_nat, on=["Year", "Week"], how="left")
for c in weather_cols:
    nat[f"{c}_anom_z"] = (nat[c] - nat["Week"].map(means[c])) / nat["Week"].map(stds[c])
if nat[anom_cols].isna().any().any():
    raise ValueError("Missing or undefined national weather anomalies.")

# Predict national deaths and construct national survivorship
nat["Expected_national"] = model.predict(nat, offset=np.log(nat["Population"]))
nat["qx_nat"] = (nat["Expected_national"] / nat["Population"]).clip(0, 1)
nat = nat.sort_values(["Year", "Week", "Age"])
nat["lx_nat"] = nat.groupby(["Year", "Week"])["qx_nat"].transform(lambda q: (1 - q).cumprod().shift(1, fill_value=1))

# Apply Brass transformation to each departmental life table
nat_lx = nat[["Year", "Week", "Age", "lx_nat"]]

dep = dep.rename(columns={"Total": "Population"}).merge(nat_lx, on=["Year", "Week", "Age"], how="left").merge(brass[["Departement", "alpha", "beta"]], on="Departement", how="left")

if dep[["lx_nat", "alpha", "beta"]].isna().any().any():
    raise ValueError("Missing national survivorship or Brass coefficients.")

eps = 1e-10
lx = dep["lx_nat"].clip(eps, 1 - eps)
dep["lx_dep"] = 1 / (1 + np.exp(-(dep["alpha"] + dep["beta"] * np.log(lx / (1 - lx)))))
dep.loc[dep["Age"] == 20, "lx_dep"] = 1.0
dep = dep.sort_values(["Departement", "Year", "Week", "Age"])
dep["lx_next"] = dep.groupby(["Departement", "Year", "Week"])["lx_dep"].shift(-1)
dep["qx_dep"] = 1 - dep["lx_next"] / dep["lx_dep"]
dep["Expected_brass"] = dep["Population"] * dep["qx_dep"]
dep_lx = dep[["Departement", "Year", "Week", "Age", "lx_dep"]]

# Evaluate ages 20-85; age 86 is only needed to calculate qx at age 85
dep_eval = dep[dep["Age"].between(20, 85)].copy()
observed = deaths.query("Year in @test_years and Week in @training_weeks and Age >= 20 and Age <= 85")
observed = observed.groupby(["Year", "Week", "Age", "Departement"], as_index=False)["Deaths"].sum()
dep_eval = dep_eval.merge(observed, on=["Year", "Week", "Age", "Departement"], how="left")
dep_eval["Deaths"] = dep_eval["Deaths"].fillna(0)

# National evaluation, aggregated over the same age range
nat_eval = nat[nat["Age"].between(20, 85)].groupby(["Year", "Week"], as_index=False).agg(Predicted=("Expected_national", "sum"))
nat_obs = observed.groupby(["Year", "Week"], as_index=False)["Deaths"].sum()
nat_eval = nat_eval.merge(nat_obs, on=["Year", "Week"], how="left").fillna({"Deaths": 0})

# Compare weekly totals and annual totals
def metrics(g, deaths_col, pred_col):
    obs, pred = g[deaths_col].to_numpy(), g[pred_col].to_numpy()
    return pd.Series({
        "Observed": obs.sum(), "Predicted": pred.sum(),
        "Bias": pred.sum() - obs.sum(),
        "Bias_pct": 100 * (pred.sum() / obs.sum() - 1) if obs.sum() else np.nan,
        "MAE_weekly": mean_absolute_error(obs, pred),
        "RMSE_weekly": np.sqrt(mean_squared_error(obs, pred))
    })

national_metrics = nat_eval.groupby("Year").apply(lambda g: metrics(g, "Deaths", "Predicted")).reset_index()
dep_weekly = dep_eval.groupby(["Departement", "Year", "Week"], as_index=False).agg(Observed=("Deaths", "sum"), Predicted=("Expected_brass", "sum"))

brass_national_weekly = dep_weekly.groupby(["Year", "Week"], as_index=False).agg(Observed=("Observed", "sum"), Predicted=("Predicted", "sum"))
brass_metrics = brass_national_weekly.groupby("Year").apply(lambda g: metrics(g, "Observed", "Predicted")).reset_index()

print("\nNational model:")
print(national_metrics.round(2).to_string(index=False))
print("\nBrass departmental predictions, aggregated nationally:")
print(brass_metrics.round(2).to_string(index=False))

brass["Departement"] = brass["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

years = pop_dep["Year"].isin(training_years)
weeks = pop_dep["Week"].isin(training_weeks)
ages = pop_dep["Age"].between(20, 86)

pop_d = pop_dep.loc[years & weeks & ages].groupby(["Departement", "Age"], as_index=False)["Total"].sum().rename(columns={"Total": "Population"})
d = deaths.loc[deaths["Year"].isin(training_years) & deaths["Week"].isin(training_weeks) & deaths["Age"].between(20, 86)].groupby(["Departement", "Age"], as_index=False)["Deaths"].sum()

obs = pop_d.merge(d, on=["Departement", "Age"], how="left").fillna({"Deaths": 0})
obs = obs.sort_values(["Departement", "Age"])
obs["qx"] = (obs["Deaths"] / obs["Population"]).clip(0, 1)
obs["lx"] = obs.groupby("Departement")["qx"].transform(lambda q: (1 - q).cumprod().shift(1, fill_value=1))

pop_n = pop_nat.loc[pop_nat["Year"].isin(training_years) & pop_nat["Week"].isin(training_weeks) & pop_nat["Age"].between(20, 86)].groupby("Age", as_index=False)["Total"].sum().rename(columns={"Total": "Population"})

deaths_n = d.groupby("Age", as_index=False)["Deaths"].sum()
nat_life = pop_n.merge(deaths_n, on="Age", how="left").fillna({"Deaths": 0})
nat_life = nat_life.sort_values("Age")
nat_life["qx"] = (nat_life["Deaths"] / nat_life["Population"]).clip(0, 1)
nat_life["lxN"] = (1 - nat_life["qx"]).cumprod().shift(1, fill_value=1)

def logit(x):
    return np.log(x / (1 - x)).where((x > 0) & (x < 1))

nat_life["logit_lxN"] = logit(nat_life["lxN"])
obs = obs.merge(nat_life[["Age", "lxN", "logit_lxN"]], on="Age", how="left")
obs["logit_lxd"] = logit(obs["lx"])

available = sorted(set(obs["Departement"]) & set(brass["Departement"]))
deps = np.random.default_rng(42).choice(available, size=min(8, len(available)), replace=False)

fig, axes = plt.subplots(len(deps), 2, figsize=(13, 3.2 * len(deps)))

for i, code in enumerate(deps):
    df = obs.loc[obs["Departement"] == code].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["logit_lxd", "logit_lxN"])
    x, y = df["logit_lxN"].to_numpy(), df["logit_lxd"].to_numpy()

    beta_fit, alpha_fit = np.polyfit(x, y, 1)
    alpha_saved, beta_saved = brass.loc[
        brass["Departement"] == code, ["alpha", "beta"]
    ].iloc[0]

    df["Saved fit"] = alpha_saved + beta_saved * x
    df["Refitted line"] = alpha_fit + beta_fit * x

    axes[i, 0].plot(df["Age"], y, "o-", label="Observed")
    axes[i, 0].plot(df["Age"], df["Saved fit"], "--", label="Saved coefficients")
    axes[i, 0].plot(df["Age"], df["Refitted line"], "--", label="Refitted coefficients")
    axes[i, 0].set(title=f"Department {code}: by age", xlabel="Age", ylabel="logit(lx)")
    axes[i, 0].legend()
    axes[i, 0].grid(alpha=0.3)

    order = np.argsort(x)
    axes[i, 1].scatter(x, y, s=18, label="Observed")
    axes[i, 1].plot(x[order], df["Saved fit"].to_numpy()[order], "--", label="Saved coefficients")
    axes[i, 1].plot(x[order], df["Refitted line"].to_numpy()[order], "--", label="Refitted line")
    axes[i, 1].set(title=f"Department {code}: Brass relationship", xlabel="logit(lx national)", ylabel="logit(lx department)")
    axes[i, 1].legend()
    axes[i, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
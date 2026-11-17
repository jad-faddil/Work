import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

base = "Life_tables/Life_tables_data/"

# 1. Load age-level predictions
nat_age = pd.read_parquet(base + "national_mortality_predictions.parquet")
brass_age = pd.read_parquet(base + "brass_mortality_predictions.parquet")
glm_age = pd.read_parquet(base + "dep_glm_age_predictions.parquet")

for df in (brass_age, glm_age):
    df["Departement"] = df["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

# Standardize column names

brass_clean = brass_age[["Departement", "Year", "Week", "Age", "Population", "Deaths", "Expected_brass"]].rename(columns={"Expected_brass": "Pred_Brass"})
glm_clean = glm_age[["Departement", "Year", "Week", "Age", "Expected_weather"]].rename(columns={"Expected_weather": "Pred_GLM"})
nat_clean = nat_age[["Year", "Week", "Age", "Population", "Expected_national"]].rename(columns={"Population": "Nat_Population", "Expected_national": "Pred_Nat"})

# Merge into unified age-level comparison DataFrame
comp_age = brass_clean.merge(glm_clean, on=["Departement", "Year", "Week", "Age"], how="inner")
comp_age = comp_age.merge(nat_clean, on=["Year", "Week", "Age"], how="left")

# Population-proportional National GLM baseline for each department-age cell
comp_age["Pred_Nat_Scaled"] = comp_age["Pred_Nat"] * (comp_age["Population"] / comp_age["Nat_Population"])

# 2. Define Age Brackets
bins = [19, 44, 64, 74, 80, 86]
labels = ["20–44", "45–64", "65–74", "75–79", "80–85"]
comp_age["Age_Group"] = pd.cut(comp_age["Age"], bins=bins, labels=labels)


# 3. Metric Helper Function
def compute_metrics(df, obs_col, pred_col):
    obs = df[obs_col].to_numpy()
    pred = df[pred_col].to_numpy()
    obs_sum, pred_sum = obs.sum(), pred.sum()
    return pd.Series({
        "Observed": obs_sum,
        "Predicted": pred_sum,
        "Bias": pred_sum - obs_sum,
        "Bias_pct": 100 * (pred_sum - obs_sum) / obs_sum if obs_sum > 0 else np.nan,
        "MAE_cell": mean_absolute_error(obs, pred),
        "RMSE_cell": np.sqrt(mean_squared_error(obs, pred))
    })


# 4. Evaluation by Age Bracket
bracket_rows = []
for grp, g in comp_age.groupby("Age_Group", observed=False):
    for label, col in [("National (Scaled)", "Pred_Nat_Scaled"), ("Brass Model", "Pred_Brass"), ("Dept GLM", "Pred_GLM")]:
        m = compute_metrics(g, "Deaths", col)
        m["Age_Group"] = grp
        m["Model"] = label
        bracket_rows.append(m)

bracket_summary = pd.DataFrame(bracket_rows)[["Age_Group", "Model", "Observed", "Predicted", "Bias_pct", "MAE_cell", "RMSE_cell"]]
print("=== PERFORMANCE BY BROAD AGE GROUP (POOLED 2025–2026) ===")
print(bracket_summary.round(2).to_string(index=False))


# 5. Granular Evaluation by Single Age (Aggregated nationally to isolate age curvature)
age_nat = comp_age.groupby(["Age"], as_index=False).agg(Observed=("Deaths", "sum"), Pred_Brass=("Pred_Brass", "sum"), Pred_GLM=("Pred_GLM", "sum"), Pred_Nat=("Pred_Nat_Scaled", "sum"))

age_nat["Bias_pct_Brass"] = 100 * (age_nat["Pred_Brass"] - age_nat["Observed"]) / age_nat["Observed"]
age_nat["Bias_pct_GLM"] = 100 * (age_nat["Pred_GLM"] - age_nat["Observed"]) / age_nat["Observed"]
age_nat["Bias_pct_Nat"] = 100 * (age_nat["Pred_Nat"] - age_nat["Observed"]) / age_nat["Observed"]


# 6. Diagnostic Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot A: Total Expected vs. Observed Deaths by Single Age
axes[0, 0].plot(age_nat["Age"], age_nat["Observed"], "k-o", label="Observed", markersize=3)
axes[0, 0].plot(age_nat["Age"], age_nat["Pred_GLM"], "b--", label="Dept GLM")
axes[0, 0].plot(age_nat["Age"], age_nat["Pred_Brass"], "r--", label="Brass")
axes[0, 0].plot(age_nat["Age"], age_nat["Pred_Nat"], "g-.", label="National GLM", alpha=0.7)
axes[0, 0].set_title("Total Mortality Curve Across Ages (2025–2026)")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Total Test Deaths")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot B: Percentage Bias by Single Age
axes[0, 1].plot(age_nat["Age"], age_nat["Bias_pct_GLM"], "b-", label="Dept GLM")
axes[0, 1].plot(age_nat["Age"], age_nat["Bias_pct_Brass"], "r-", label="Brass")
axes[0, 1].plot(age_nat["Age"], age_nat["Bias_pct_Nat"], "g-.", label="National GLM", alpha=0.7)
axes[0, 1].axhline(0, color="k", linestyle="--", alpha=0.6)
axes[0, 1].set_title("Percentage Bias across Single Ages (%)")
axes[0, 1].set_xlabel("Age")
axes[0, 1].set_ylabel("Bias (%) = 100 * (Pred - Obs) / Obs")
axes[0, 1].set_ylim(-40, 40)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot C: Cell-level RMSE across Single Ages
age_rmse = comp_age.groupby("Age").apply(
    lambda g: pd.Series({
        "RMSE_GLM": np.sqrt(mean_squared_error(g["Deaths"], g["Pred_GLM"])),
        "RMSE_Brass": np.sqrt(mean_squared_error(g["Deaths"], g["Pred_Brass"])),
        "RMSE_Nat": np.sqrt(mean_squared_error(g["Deaths"], g["Pred_Nat_Scaled"]))
    }), include_groups=False).reset_index()

axes[1, 0].plot(age_rmse["Age"], age_rmse["RMSE_GLM"], "b-", label="Dept GLM")
axes[1, 0].plot(age_rmse["Age"], age_rmse["RMSE_Brass"], "r-", label="Brass")
axes[1, 0].plot(age_rmse["Age"], age_rmse["RMSE_Nat"], "g-.", label="National GLM", alpha=0.7)
axes[1, 0].set_title("Weekly Department-Cell RMSE by Age")
axes[1, 0].set_xlabel("Age")
axes[1, 0].set_ylabel("RMSE")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot D: Age Bracket RMSE Bar Chart
bracket_rmse = bracket_summary.pivot(index="Age_Group", columns="Model", values="RMSE_cell")
bracket_rmse[["National (Scaled)", "Brass Model", "Dept GLM"]].plot(kind="bar", ax=axes[1, 1], colormap="viridis", alpha=0.85, rot=0)
axes[1, 1].set_title("RMSE by Demographic Age Bracket")
axes[1, 1].set_ylabel("RMSE per Department-Week Cell")
axes[1, 1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
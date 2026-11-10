import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

base = "Life_tables/Life_tables_data/"

# 1. Load predictions
nat_pred = pd.read_parquet(base + "national_mortality_predictions.parquet")
brass_pred = pd.read_parquet(base + "brass_mortality_predictions.parquet")
glm_weekly = pd.read_parquet(base + "dep_glm_predictions.parquet")

for df in (brass_pred, glm_weekly):
    df["Departement"] = df["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

# Aggregate Brass age-level data to weekly departmental totals
brass_weekly = brass_pred.groupby(["Departement", "Year", "Week"], as_index=False).agg(Observed=("Deaths", "sum"), Population=("Population", "sum"), Pred_Brass=("Expected_brass", "sum"), Pred_Brass_Normal=("Expected_brass_normal", "sum"))

# Aggregate National GLM predictions to weekly national totals
nat_weekly_total = nat_pred.groupby(["Year", "Week"], as_index=False).agg(Pred_Nat_Direct=("Expected_national", "sum"), Nat_Population=("Population", "sum"), Pred_Nat_Normal=("Expected_national_normal", "sum"))

# Merge Departmental GLM and Brass
comp = glm_weekly.rename(columns={"Expected_deps": "Pred_GLM","Expected_deps_normal": "Pred_GLM_Normal"})[["Departement", "Year", "Week", "Observed", "Pred_GLM", "Pred_GLM_Normal"]].merge(
    brass_weekly[["Departement", "Year", "Week", "Population", "Pred_Brass", "Pred_Brass_Normal"]], on=["Departement", "Year", "Week"], how="inner")

# Create a naive baseline: distribute national GLM predictions by department population share
comp = comp.merge(nat_weekly_total, on=["Year", "Week"], how="left")
comp["Pred_Nat_Proportional"] = comp["Pred_Nat_Direct"] * (comp["Population"] / comp["Nat_Population"])
comp["Pred_Nat_Normal_Proportional"] = comp["Pred_Nat_Normal"] * (comp["Population"] / comp["Nat_Population"])


# 2. Metric calculation function
def compute_metrics(df, obs_col, pred_col, pred_normal_col):
    obs = df[obs_col].to_numpy()
    pred = df[pred_col].to_numpy()
    pred_normal = df[pred_normal_col].to_numpy()
    obs_sum, pred_sum, pred_normal_sum = obs.sum(), pred.sum(), pred_normal.sum()
    return pd.Series({
        "Observed": obs_sum,
        "Predicted": pred_sum,
        "Predicted_normal" : pred_normal_sum,
        "Bias": pred_sum - obs_sum,
        "Heat_impact": pred_sum - pred_normal_sum,
        "Bias_pct": 100 * (pred_sum - obs_sum) / obs_sum if obs_sum > 0 else np.nan,
        "MAE_weekly": mean_absolute_error(obs, pred),
        "RMSE_weekly": np.sqrt(mean_squared_error(obs, pred))
    })


# 3. National Weekly Evaluation (Macro Level)
nat_macro = comp.groupby(["Year", "Week"], as_index=False).agg(
    Observed=("Observed", "sum"), Pred_GLM_Sum=("Pred_GLM", "sum"), Pred_Brass_Sum=("Pred_Brass", "sum"), Pred_GLM_Normal_Sum=("Pred_GLM_Normal", "sum"), Pred_Brass_Normal_Sum=("Pred_Brass_Normal", "sum")
    ).merge(nat_weekly_total[["Year", "Week", "Pred_Nat_Direct", "Pred_Nat_Normal"]], on=["Year", "Week"], how="left")

nat_rows = []
for year, g in nat_macro.groupby("Year"):
    for label, col, normal_col in [("National GLM (Direct)", "Pred_Nat_Direct", "Pred_Nat_Normal"), ("Dept GLM (Aggregated)", "Pred_GLM_Sum", "Pred_GLM_Normal_Sum"), ("Brass (Aggregated)", "Pred_Brass_Sum", "Pred_Brass_Normal_Sum")]:
        m = compute_metrics(g, "Observed", col, normal_col)
        m["Year"] = year
        m["Model"] = label
        nat_rows.append(m)

nat_summary = pd.DataFrame(nat_rows)[["Year", "Model", "Observed", "Predicted", "Predicted_normal", "Bias", "Heat_impact", "Bias_pct", "MAE_weekly", "RMSE_weekly"]]
print("=== 1. NATIONAL-LEVEL WEEKLY TOTALS ===")
print(nat_summary.round(2).to_string(index=False))


# 4. Departmental Weekly Evaluation (Micro Level: Pooled across all 95 departments)
dep_rows = []
for year, g in comp.groupby("Year"):
    for label, col, normal_col in [("National GLM (Pop-Proportional)", "Pred_Nat_Proportional", "Pred_Nat_Normal_Proportional"), ("Brass Model", "Pred_Brass", "Pred_Brass_Normal"), ("Dept GLM (Local Weather)", "Pred_GLM", "Pred_GLM_Normal")]:
        m = compute_metrics(g, "Observed", col, normal_col)
        m["Year"] = year
        m["Model"] = label
        dep_rows.append(m)

dep_summary = pd.DataFrame(dep_rows)[["Year", "Model", "Observed", "Predicted", "Predicted_normal", "Bias", "Heat_impact", "Bias_pct", "MAE_weekly", "RMSE_weekly"]]
print("\n=== 2. DEPARTMENTAL WEEKLY TOTALS (POOLED 95 DEPARTMENTS) ===")
print(dep_summary.round(2).to_string(index=False))


# 5. Department-by-Department Head-to-Head
dep_nat_m = comp.groupby("Departement").apply(lambda g: compute_metrics(g, "Observed", "Pred_Nat_Proportional", "Pred_Nat_Normal_Proportional"), include_groups=False).reset_index()
dep_brass_m = comp.groupby("Departement").apply(lambda g: compute_metrics(g, "Observed", "Pred_Brass", "Pred_Brass_Normal"), include_groups=False).reset_index()
dep_glm_m = comp.groupby("Departement").apply(lambda g: compute_metrics(g, "Observed", "Pred_GLM", "Pred_GLM_Normal"), include_groups=False).reset_index()

h2h = dep_glm_m[["Departement", "Observed", "RMSE_weekly", "MAE_weekly", "Bias_pct"]].rename(
    columns={"RMSE_weekly": "RMSE_GLM", "MAE_weekly": "MAE_GLM", "Bias_pct": "Bias_pct_GLM"}
).merge(dep_brass_m[["Departement", "RMSE_weekly", "MAE_weekly", "Bias_pct"]].rename(columns={"RMSE_weekly": "RMSE_Brass", "MAE_weekly": "MAE_Brass", "Bias_pct": "Bias_pct_Brass"}), on="Departement"
).merge(dep_nat_m[["Departement", "RMSE_weekly", "MAE_weekly", "Bias_pct"]].rename(columns={"RMSE_weekly": "RMSE_Nat", "MAE_weekly": "MAE_Nat", "Bias_pct": "Bias_pct_Nat"}), on="Departement")

print("\n=== 3. DEPARTMENTAL WIN COUNT (LOWEST METRIC) ===")
rmse_wins = h2h[["RMSE_Brass", "RMSE_GLM"]].idxmin(axis=1).value_counts()
mae_wins = h2h[["MAE_Brass", "MAE_GLM"]].idxmin(axis=1).value_counts()
bias_wins = h2h[["Bias_pct_Brass", "Bias_pct_GLM"]].abs().idxmin(axis=1).value_counts()

print("Best Weekly RMSE:\n", rmse_wins.to_string())
print("\nBest Weekly MAE:\n", mae_wins.to_string())
print("\nLowest Absolute Bias %:\n", bias_wins.to_string())

# 6. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot A: National Weekly Trajectory
nat_macro["Time"] = nat_macro["Year"].astype(str) + "-W" + nat_macro["Week"].astype(str).str.zfill(2)
axes[0, 0].plot(nat_macro["Time"], nat_macro["Observed"], "k-o", label="Observed", markersize=4)
axes[0, 0].plot(nat_macro["Time"], nat_macro["Pred_Nat_Direct"], "g-.", label="National GLM")
axes[0, 0].plot(nat_macro["Time"], nat_macro["Pred_Brass_Sum"], "r--", label="Brass (Aggregated)")
axes[0, 0].plot(nat_macro["Time"], nat_macro["Pred_GLM_Sum"], "b--", label="Dept GLM (Aggregated)")
axes[0, 0].set_title("Aggregated Weekly Deaths in France (2025–2026)")
axes[0, 0].set_ylabel("Weekly Deaths")
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].set_xticks(axes[0, 0].get_xticks()[::2])
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot B: RMSE Distribution across 95 Departments
axes[0, 1].boxplot([h2h["RMSE_Nat"], h2h["RMSE_Brass"], h2h["RMSE_GLM"]], tick_labels=["National (Scaled)", "Brass Model", "Dept GLM"], patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.7))
axes[0, 1].set_title("Departmental Weekly RMSE Distribution")
axes[0, 1].set_ylabel("Weekly RMSE")
axes[0, 1].grid(alpha=0.3)

# Plot C: Bias % Distribution across 95 Departments
bins = np.linspace(-30, 30, 25)
axes[1, 0].hist(h2h["Bias_pct_Nat"], bins=bins, alpha=0.4, label="National (Scaled)", color="green")
axes[1, 0].hist(h2h["Bias_pct_Brass"], bins=bins, alpha=0.5, label="Brass Model", color="red")
axes[1, 0].hist(h2h["Bias_pct_GLM"], bins=bins, alpha=0.5, label="Dept GLM", color="blue")
axes[1, 0].axvline(0, color="k", linestyle="--", alpha=0.7)
axes[1, 0].set_title("Distribution of Departmental Bias (%)")
axes[1, 0].set_xlabel("Bias (%) = 100 * (Predicted - Observed) / Observed")
axes[1, 0].set_ylabel("Number of Departments")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot D: Department RMSE: Brass vs. Dept GLM (with National reference point)
max_val = max(h2h["RMSE_GLM"].max(), h2h["RMSE_Brass"].max()) * 1.05
axes[1, 1].scatter(h2h["RMSE_GLM"], h2h["RMSE_Brass"], c=h2h["Observed"], cmap="viridis", alpha=0.8, s=35)
axes[1, 1].plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="1:1 Equality Line")
cbar = plt.colorbar(axes[1, 1].scatter(h2h["RMSE_GLM"], h2h["RMSE_Brass"], c=h2h["Observed"], cmap="viridis", alpha=0.8, s=35), ax=axes[1, 1])
cbar.set_label("Observed Departmental Deaths")
axes[1, 1].set_title("Department Weekly RMSE: Dept GLM vs. Brass")
axes[1, 1].set_xlabel("Dept GLM RMSE")
axes[1, 1].set_ylabel("Brass RMSE")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
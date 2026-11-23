import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

base = "Life_tables/Life_tables_data/"

# 1. Load predictions
glm_weekly = pd.read_parquet(base + "dep_glm_predictions.parquet")
glm_weekly["Departement"] = glm_weekly["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
comp = glm_weekly.rename(columns={"Expected_deps": "Pred_GLM","Expected_deps_normal": "Pred_GLM_Normal"})[["Departement", "Year", "Week", "Observed", "Pred_GLM", "Pred_GLM_Normal"]]

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
nat_macro = comp.groupby(["Year", "Week"], as_index=False).agg(Observed=("Observed", "sum"), Pred_GLM_Sum=("Pred_GLM", "sum"), Pred_GLM_Normal_Sum=("Pred_GLM_Normal", "sum"))

nat_rows = []
for year, g in nat_macro.groupby("Year"):
    m = compute_metrics(g, "Observed", "Pred_GLM_Sum", "Pred_GLM_Normal_Sum")
    m["Year"] = year
    m["Model"] = "Dept GLM"
    nat_rows.append(m)

nat_summary = pd.DataFrame(nat_rows)[["Year", "Model", "Observed", "Predicted", "Predicted_normal", "Bias", "Heat_impact", "Bias_pct", "MAE_weekly", "RMSE_weekly"]]
print("=== 1. NATIONAL-LEVEL WEEKLY TOTALS ===")
print(nat_summary.round(2).to_string(index=False))

# 4. Departmental Weekly Evaluation (Micro Level: Pooled across all 95 departments)
dep_rows = []
for year, g in comp.groupby("Year"):
    m = compute_metrics(g, "Observed", "Pred_GLM", "Pred_GLM_Normal")
    m["Year"] = year
    m["Model"] = "Dept GLM"
    dep_rows.append(m)

dep_summary = pd.DataFrame(dep_rows)[["Year", "Model", "Observed", "Predicted", "Predicted_normal", "Bias", "Heat_impact", "Bias_pct", "MAE_weekly", "RMSE_weekly"]]
print("\n=== 2. DEPARTMENTAL WEEKLY TOTALS (POOLED 95 DEPARTMENTS) ===")
print(dep_summary.round(2).to_string(index=False))

# 5. Visualizations
dep_glm_m = comp.groupby("Departement").apply(lambda g: compute_metrics(g, "Observed", "Pred_GLM", "Pred_GLM_Normal"), include_groups=False).reset_index()
h2h = dep_glm_m[["Departement", "Observed", "RMSE_weekly", "MAE_weekly", "Bias_pct"]].rename(
    columns={"RMSE_weekly": "RMSE_GLM", "MAE_weekly": "MAE_GLM", "Bias_pct": "Bias_pct_GLM"})

# Plot A: National Weekly Trajectory
figure = plt.figure(figsize=(10, 6))
nat_macro["Time"] = nat_macro["Year"].astype(str) + "-W" + nat_macro["Week"].astype(str).str.zfill(2)
plt.plot(nat_macro["Time"], nat_macro["Observed"], "k-o", label="Observed", markersize=4)
plt.plot(nat_macro["Time"], nat_macro["Pred_GLM_Sum"], "b--", label="Dept GLM (Aggregated)")
plt.title("Aggregated Weekly Deaths in France (2025–2026)")
plt.ylabel("Weekly Deaths")
plt.tick_params(axis="x", rotation=45)
plt.xticks()
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Plot B: Bias % Distribution across 95 Departments
figure = plt.figure(figsize=(10, 6))
bins = np.linspace(-30, 30, 25)
plt.hist(h2h["Bias_pct_GLM"], bins=bins, alpha=0.5, label="Dept GLM", color="blue")
plt.axvline(0, color="k", linestyle="--", alpha=0.7)
plt.title("Distribution of Departmental Bias (%)")
plt.xlabel("Bias (%) = 100 * (Predicted - Observed) / Observed")
plt.ylabel("Number of Departments")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
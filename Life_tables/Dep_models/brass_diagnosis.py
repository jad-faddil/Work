
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = "Life_tables/Life_tables_data/"
nat = pd.read_parquet(base + "national_mortality_predictions.parquet")
dep = pd.read_parquet(base + "brass_mortality_predictions.parquet")
deaths = pd.read_parquet(base + "deaths_weekly.parquet")

print("National columns:", nat.columns.tolist())
print("Departmental columns:", dep.columns.tolist())

# Standardize the prediction column names if needed.
nat = nat.rename(columns={"Expected_national": "Predicted_national", "Expected": "Predicted_national"})
dep = dep.rename(columns={"Expected_brass": "Predicted_dep", "Expected": "Predicted_dep"})

# 1. Compare national prediction vs sum of departmental predictions by age and year.
nat_age = nat.query("Year in [2025, 2026] and Week >= 22 and Week <= 36").groupby(["Year", "Age"], as_index=False)["Predicted_national"].sum()
dep_age = dep.query("Year in [2025, 2026] and Week >= 22 and Week <= 36").groupby(["Year", "Age"], as_index=False)["Predicted_dep"].sum()

age_comparison = nat_age.merge(dep_age, on=["Year", "Age"], how="outer")
age_comparison["Difference"] = age_comparison["Predicted_dep"] - age_comparison["Predicted_national"]
age_comparison["Ratio"] = age_comparison["Predicted_dep"] / age_comparison["Predicted_national"]
print("\nPredicted deaths by age:")
print(age_comparison.round(2).to_string(index=False))

# 2. Find departments with the largest total predictions.
dep_total = dep.query("Year in [2025, 2026] and Week >= 22 and Week <= 36").groupby(["Year", "Departement"], as_index=False)["Predicted_dep"].sum()

deaths["Departement"] = deaths["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
deaths = deaths.query("Year in [2025, 2026] and Week >= 22 and Week <= 36 and Age >= 20 and Age <= 85")
observed_dep = deaths.groupby(["Year", "Departement"], as_index=False)["Deaths"].sum()

dep_total["Departement"] = dep_total["Departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
dep_total = dep_total.merge(observed_dep, on=["Year", "Departement"], how="left")
dep_total["Bias"] = dep_total["Predicted_dep"] - dep_total["Deaths"]

print("\nHighest departmental predictions:")
print(dep_total.sort_values(["Year", "Predicted_dep"], ascending=[True, False]).groupby("Year").head(10).to_string(index=False))


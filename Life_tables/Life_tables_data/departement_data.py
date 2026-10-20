import pandas as pd
import numpy as np

# Load data
pop = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_all.parquet")
dist = pd.read_csv("Life_tables/Life_tables_data/distribution_population_departements.csv", sep=";")

dist = dist.rename(columns={"Code": "Departement", "Proportion (%)": "Proportion"})
dist["Departement"] = dist["Departement"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
dist["Departement"] = dist["Departement"].str.zfill(2)
dist["Proportion"] = pd.to_numeric(dist["Proportion"], errors="coerce")

# Merge departments 2A and 2B into department 20
dist["Departement"] = dist["Departement"].replace({"2A": "20", "2B": "20"})
dist = dist.groupby("Departement", as_index=False)["Proportion"].sum()

# Remove overseas departments
excluded = ["971", "972", "973", "974", "976"]
dist = dist[~dist["Departement"].isin(excluded)].copy()

# Renormalize proportions to sum to 100%
dist["Proportion"] /= dist["Proportion"].sum()

# Cross merge: one row per week, age, and department
result = pop.merge(dist[["Departement", "Proportion"]], how="cross")

# Allocate national population to departments
for col in ["Female", "Male", "Total"]:
    result[col] *= result["Proportion"]

result = result.drop(columns="Proportion")

# Validate that departmental populations sum to national totals
keys = ["Year", "Week", "Age"]
cols = ["Female", "Male", "Total"]

check = result.groupby(keys)[cols].sum().sort_index()
target = pop.set_index(keys)[cols].sort_index()

assert np.allclose(check, target, rtol=1e-5, atol=1e-5), \
    "Departmental population totals do not match national totals!"

# Save
result.to_parquet("Life_tables/Life_tables_data/weekly_population_departements.parquet", index=False)

print(result.head())
print("Shape:", result.shape)
print("Number of departments:", result["Departement"].nunique())
print("Normalized proportion sum:", dist["Proportion"].sum())
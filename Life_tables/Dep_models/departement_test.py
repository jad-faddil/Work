import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
population = pd.read_parquet("Life_tables/Life_tables_data/weekly_population_departements.parquet")
deaths = pd.read_parquet("Life_tables/Life_tables_data/deaths_weekly.parquet")

# Aggregate deaths across any remaining dimensions, e.g. sex
keys = ["Year", "Week", "Age", "Departement"]
deaths = deaths.groupby(keys, as_index=False)["Deaths"].sum()

# Build departmental dataset
population = population[keys + ["Total"]].rename(columns={"Total": "Population"})
data = population.merge(deaths, on=keys, how="left")
data["Deaths"] = data["Deaths"].fillna(0).astype(int)
data["Population"] = data["Population"].round().astype(int)

# Restrict to the same ages and weeks as the national model
data = data[data["Age"].between(20, 85) & data["Week"].between(22, 36)].copy()

# Baseline mortality rate by department and age, estimated over 2019–2024
base = (data[data["Year"].between(2019, 2024)].groupby(["Departement", "Age"], as_index=False).agg(Deaths=("Deaths", "sum"), Population=("Population", "sum")))
base["qbase"] = base["Deaths"] / base["Population"]

data = data.merge(base[["Departement", "Age", "qbase"]], on=["Departement", "Age"], how="left",validate="many_to_one")
print("Department-age cells with zero baseline mortality:", (base["qbase"] == 0).sum())

# Fit one seasonal model per department
all_multipliers = []
results = []

for dep, dept_data in data.groupby("Departement", sort=True):
    train = dept_data[dept_data["Year"].between(2019, 2024)].copy()
    train = train[(train["qbase"] > 0) & (train["Population"] > 0)].copy()
    train["Week"] = train["Week"].astype("int64")

    seasonal_model = smf.glm("Deaths ~ 0 + C(Week)",data=train,family=sm.families.Poisson(),offset=np.log(train["Population"] * train["qbase"])).fit()

    seasonal_multiplier = pd.Series({int(term.rsplit("[", 1)[1].rstrip("]")): np.exp(coef)for term, coef in seasonal_model.params.items()}, name="Seasonal_multiplier")

    expected_weeks = set(range(22, 37))
    if set(seasonal_multiplier.index) != expected_weeks:
        raise ValueError(f"Missing seasonal week coefficients for department {dep}")

    mult = seasonal_multiplier.rename_axis("Week").reset_index()
    mult["Departement"] = dep
    all_multipliers.append(mult[["Departement", "Week", "Seasonal_multiplier"]])

    # Evaluate 2025 and 2026
    for year in [2025, 2026]:
        test = dept_data[dept_data["Year"] == year].copy()
        test["Seasonal_multiplier"] = test["Week"].map(seasonal_multiplier)

        if test["Seasonal_multiplier"].isna().any():
            raise ValueError(f"Missing multiplier in department {dep}, year {year}")

        test["Expected_baseline"] = test["Population"] * test["qbase"]
        test["Expected_seasonal"] = test["Expected_baseline"] * test["Seasonal_multiplier"]

        observed = test["Deaths"].sum()
        expected_base = test["Expected_baseline"].sum().astype(int)
        expected_seasonal = test["Expected_seasonal"].sum().astype(int)

        results.append({
            "Departement": dep,
            "Year": year,
            "Observed": observed,
            "Expected_baseline": expected_base,
            "Excess_baseline": observed - expected_base,
            "Excess_baseline_pct": 100 * (observed / expected_base - 1) if expected_base > 0 else np.nan,
            "Expected_seasonal": expected_seasonal,
            "Excess_seasonal": observed - expected_seasonal,
            "Excess_seasonal_pct": 100 * (observed / expected_seasonal - 1) if expected_seasonal > 0 else np.nan
        })

# Save results
multipliers = pd.concat(all_multipliers, ignore_index=True)
summary = pd.DataFrame(results)

print("\nOverall results by year:")
print(summary.groupby("Year")[["Observed", "Expected_baseline", "Expected_seasonal", "Excess_baseline", "Excess_seasonal"]].sum().round(1))

print("\nDepartmental results (first 10 rows):")
print(summary.head(10).to_string(index=False))


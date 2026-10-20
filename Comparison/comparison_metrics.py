import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_parquet("Comparison/model_comparison_weekly.parquet")
data["date"] = pd.to_datetime(data["date"])

models = ["transformer", "glm", "xgb", "tfm"]

for model in models:
    eval_data = data.dropna(subset=[model])
    mae = mean_absolute_error(eval_data["observed"], eval_data[model])
    rmse = np.sqrt(mean_squared_error(eval_data["observed"], eval_data[model]))
    print(f"{model:12s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f}")

for year in [2023, 2024]:
    year_data = data[data["date"].dt.year == year].dropna(subset=["observed"]).copy()

    print(f"\n===== {year} =====")

    for model in models:
        eval_data = year_data.dropna(subset=[model])
        mae = mean_absolute_error(eval_data["observed"], eval_data[model])
        rmse = np.sqrt(mean_squared_error(eval_data["observed"], eval_data[model]))
        print(f"{model:12s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f}")

    plt.figure(figsize=(14, 6))
    plt.plot(year_data["date"], year_data["observed"], label="Observed", linewidth=2)

    for model in models:
        plt.plot(year_data["date"], year_data[model], label=model.upper())

    plt.title(f"Weekly mortality predictions vs observed — {year}")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
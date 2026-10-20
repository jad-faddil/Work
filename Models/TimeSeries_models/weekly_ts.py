from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_parquet("Data/data_weekly.parquet")
data.rename(columns={"week": "date"}, inplace=True)

# National series for each sex × age group
temp_cols = ["TN_mean", "TN_hot_days", "TX_mean", "TX_hot_days"]

data = data.groupby(["date", "sex", "age_group"]).agg(
    deaths=("deaths", "sum"),
    exposure=("exposure", "sum"),
    **{col: (col, "mean") for col in temp_cols}
).reset_index()

data["mortality_rate"] = data["deaths"] / data["exposure"]
data["log_rate"] = np.log(data["deaths"] + 0.5) - np.log(data["exposure"])

predictions = []

for (sex, age), group in data.groupby(["sex", "age_group"]):
    print(sex, age)
    group = group.sort_values("date").set_index("date").asfreq("W-MON")

    train = group[(group.index >= "2004-01-01") & (group.index <= "2022-12-31")]
    test = group[group.index >= "2023-01-01"].copy()

    model = SARIMAX(train["log_rate"], exog=train[temp_cols], order=(1, 1, 1), seasonal_order=(1, 0, 1, 52)).fit(disp=False, maxiter=5000)
    forecast = model.forecast(steps=len(test), exog=test[temp_cols])

    test["prediction"] = np.exp(forecast)
    test["pred_deaths"] = test["prediction"] * test["exposure"]

    test["sex"] = sex
    test["age_group"] = age

    predictions.append(test.reset_index())

pred = pd.concat(predictions, ignore_index=True)
pred["pred_deaths"] = pred["prediction"] * pred["exposure"]

mae = np.mean(np.abs(pred["deaths"] - pred["pred_deaths"]))
rmse = np.sqrt(np.mean((pred["deaths"] - pred["pred_deaths"]) ** 2))

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

observed = (data[data["date"] >= "2023-01-01"].groupby("date")["deaths"].sum())
national = pred.groupby("date")["pred_deaths"].sum()

plt.figure(figsize=(14, 6))
plt.plot(observed.index, observed.values, label="Observed")
plt.plot(national.index, national.values, label="SARIMA")
plt.xlabel("Date")
plt.ylabel("Weekly deaths")
plt.title("ARIMA predictions vs observed mortality")
plt.legend()
plt.tight_layout()
plt.show()
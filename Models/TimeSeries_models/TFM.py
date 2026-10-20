from timesfm3 import TimesFM3Forecaster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load granular mortality data
df = pd.read_parquet("Data/data_weekly.parquet")
df["date"] = pd.to_datetime(df["date"])

# Aggregate over departments
mortality = df.groupby(["date", "sex", "age_group"], as_index=False).agg(
    deaths=("deaths", "sum"), exposure=("exposure", "sum"))

mortality["mortality_rate"] = mortality["deaths"] / mortality["exposure"]
ts = mortality.pivot(index="date", columns=["sex", "age_group"], values="mortality_rate").sort_index()

# Exposure-weighted weekly TM for each sex × age series
df["TM_weighted"] = df["TM"] * df["exposure"]

temp = df.groupby(["date", "sex", "age_group"], as_index=False).agg(
    TM_weighted=("TM_weighted", "sum"),
    exposure=("exposure", "sum")
)
temp["TM"] = temp["TM_weighted"] / temp["exposure"]

tm = temp.pivot(index="date", columns=["sex", "age_group"], values="TM").reindex(ts.index)

model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

train = ts.loc[(ts.index < "2023-01-01") & (ts.index >= "2004-01-01")]
test = ts.loc[ts.index >= "2023-01-01"]

# TimesFM expects (variates, time)
tm_train = tm.loc[train.index]
tm_test = tm.loc[test.index]
context = train.values.T.astype(np.float32)
horizon = 105

tm_context = tm_train.values.T.astype(np.float32)
tm_future = tm_test.iloc[:horizon].values.T.astype(np.float32)
tm_covariate = np.concatenate([tm_context, tm_future], axis=1)

output_base = model.predict(context=context, horizon=horizon)
forecast_base = output_base.forecast

output = model.predict(context=context, horizon=horizon, past_future_covariates=tm_covariate)
forecast = output.forecast

forecast_dates = test.index[:horizon]

pred_base = pd.DataFrame(forecast_base.T, index=forecast_dates, columns=ts.columns)
pred_temp = pd.DataFrame(forecast.T, index=forecast_dates, columns=ts.columns)
actual = test.loc[forecast_dates]

for name, pred in [("No temperature", pred_base), ("Temperature", pred_temp)]:
    error = pred - actual
    mae = np.abs(error).mean().mean()
    rmse = np.sqrt((error ** 2).mean().mean())

    print(f"\n{name}")
    print(f"MAE:  {mae:.6e}")
    print(f"RMSE: {rmse:.6e}")

# Side-by-side plots
for year in [2023, 2024]:
    d_actual = actual[actual.index.year == year]
    d_base = pred_base[pred_base.index.year == year]
    d_temp = pred_temp[pred_temp.index.year == year]

    fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True)

    for i, age in enumerate(ts.columns.get_level_values("age_group").unique()):
        ax = axes.flat[i]

        for sex in ["F", "M"]:
            col = (sex, age)

            ax.plot(d_actual.index, d_actual[col], label=f"{sex} actual")
            ax.plot(d_base.index, d_base[col], "--", label=f"{sex} no temp")
            ax.plot(d_temp.index, d_temp[col], ":", label=f"{sex} + temp")

        ax.set_title(f"Age {age}")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.3)

    axes[0, 0].legend()
    fig.suptitle(f"TimesFM — {year} weekly mortality-rate forecast", fontsize=16)
    plt.tight_layout()
    plt.show()
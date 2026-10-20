import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_parquet("Data/data_weekly.parquet")
data.rename(columns={"week": "date"}, inplace=True)

# National series for each sex × age group
temp_cols = [
    "TN_mean", "TN_hot_days",
    "TX_mean", "TX_hot_days"
]

data = data.groupby(["date", "sex", "age_group"]).agg(
    deaths=("deaths", "sum"),
    exposure=("exposure", "sum"),
    **{col: (col, "mean") for col in temp_cols}
).reset_index()

data["mortality_rate"] = data["deaths"] / data["exposure"]
data["log_rate"] = np.log(data["deaths"] + 0.5) - np.log(data["exposure"])
data["lag1"] = data.groupby(["sex", "age_group"])["log_rate"].shift(1)

# Lag 1 within each of the 18 series
data["lag1"] = data.groupby(["sex", "age_group"])["mortality_rate"].shift(1)

train = data[(data["date"] >= "2004-01-01") & (data["date"] <= "2022-12-31")].copy()
test = data[data["date"] >= "2023-01-01"].copy()

scaler = StandardScaler()

train[temp_cols] = scaler.fit_transform(train[temp_cols])
test[temp_cols] = scaler.transform(test[temp_cols])

class TempNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(temp_cols), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return 20 * torch.tanh(self.net(x)).squeeze(-1)
    
predictions = []

for (sex, age), group in train.groupby(["sex", "age_group"]):
    group = group.dropna(subset=["lag1"] + temp_cols)

    lag = torch.tensor(group["lag1"].values, dtype=torch.float32, device=device)
    X = torch.tensor(group[temp_cols].values, dtype=torch.float32, device=device)
    y = torch.tensor((group["log_rate"] - group["lag1"]).values, dtype=torch.float32, device=device)

    model = TempNN().to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(20000):
        optimizer.zero_grad()

        prediction = model(X)
        loss = loss_fn(prediction, y)

        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f"{sex} {age} | Epoch {epoch + 1}/20000 | Loss: {loss.item():.6f}")

    test_group = test[(test["sex"] == sex) & (test["age_group"] == age)].copy()
    test_group = test_group.sort_values("date").reset_index(drop=True)

    previous = group.iloc[-1]["log_rate"]
    preds = []

    X_test = torch.tensor(test_group[temp_cols].values, dtype=torch.float32, device=device)

    with torch.no_grad():
        temp_effect = model(X_test).cpu().numpy()

    for i in range(len(test_group)):
        prediction = previous + temp_effect[i]
        preds.append(np.exp(prediction))
        previous = prediction

    test_group["prediction"] = preds
    predictions.append(test_group)

test_pred = pd.concat(predictions, ignore_index=True)

for (sex, age), group in test_pred.groupby(["sex", "age_group"]):
    print(
        sex, age,
        "MAE:", mean_absolute_error(group["mortality_rate"], group["prediction"]),
        "RMSE:", np.sqrt(mean_squared_error(group["mortality_rate"], group["prediction"])),
        "R2:", r2_score(group["mortality_rate"], group["prediction"])
    )

print("\nOverall:")
print(f"MAE:  {mean_absolute_error(test_pred['mortality_rate'], test_pred['prediction']):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(test_pred['mortality_rate'], test_pred['prediction'])):.4f}")
print(f"R2:   {r2_score(test_pred['mortality_rate'], test_pred['prediction']):.4f}")

test_pred["residual"] = test_pred["mortality_rate"] - test_pred["prediction"]

sigma = (test_pred.groupby(["sex", "age_group"])["residual"].std().reset_index(name="sigma"))

test_pred = test_pred.merge(sigma, on=["sex", "age_group"])

test_pred["lower"] = test_pred["prediction"] - 1.96 * test_pred["sigma"]
test_pred["upper"] = test_pred["prediction"] + 1.96 * test_pred["sigma"]

for (sex, age), group in test_pred.groupby(["sex", "age_group"]):
    group = group.sort_values("date")

    plt.figure(figsize=(14, 5))

    plt.plot(group["date"], group["mortality_rate"], label="Observed", linewidth=1.5)
    plt.plot(group["date"], group["prediction"], label="AR(1)", linewidth=1.5)
    plt.fill_between(group["date"], group["lower"],group["upper"], alpha=0.2, label="95% interval")
    plt.yscale("log")
    plt.title(f"{sex} — {age}")
    plt.xlabel("Date")
    plt.ylabel("Mortality rate")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    break
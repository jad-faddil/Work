import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

class MLP(nn.Module):
    def __init__(self, d_input, d_hidden_1, d_hidden_2):
        super().__init__()
        self.layer1 = nn.Linear(d_input, d_hidden_1)
        self.layer2 = nn.Linear(d_hidden_1, d_hidden_2)
        self.output = nn.Linear(d_hidden_2, 1)

    def forward(self, x):
        x = nn.functional.gelu(self.layer1(x))
        x = nn.functional.gelu(self.layer2(x))
        return self.output(x).squeeze(1)



data = pd.read_parquet("Data/data_nn.parquet")
data["year"] = data["week"].dt.year

temp_cols = [f"{v}_{d}" for v in ["TN", "TM", "TX"] for d in range(1, 8)]

train = data[data["week"].between("2004-01-01", "2022-12-31")].copy()
test = data[data["week"] >= "2023-01-01"].copy()

# GLM features
week_num_train = train["week"].dt.isocalendar().week.astype(float).to_numpy()
week_num_test = test["week"].dt.isocalendar().week.astype(float).to_numpy()

train["week_sin"] = np.sin(2 * np.pi * week_num_train / 52)
train["week_cos"] = np.cos(2 * np.pi * week_num_train / 52)
test["week_sin"] = np.sin(2 * np.pi * week_num_test / 52)
test["week_cos"] = np.cos(2 * np.pi * week_num_test / 52)

X_train = pd.concat([
    pd.get_dummies(train[["sex", "age_group"]], drop_first=True, dtype=float),
    train[["departement", "year", "week_sin", "week_cos"]].astype(float)
], axis=1)

X_test = pd.concat([
    pd.get_dummies(test[["sex", "age_group"]], drop_first=True, dtype=float),
    test[["departement", "year", "week_sin", "week_cos"]].astype(float)
], axis=1)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)

X_temp_train = torch.tensor(train[temp_cols].values, dtype=torch.float32)
X_temp_test = torch.tensor(test[temp_cols].values, dtype=torch.float32)

mean = X_temp_train.mean(0)
std = X_temp_train.std(0)

X_temp_train = (X_temp_train - mean) / std
X_temp_test = (X_temp_test - mean) / std

exposure_train = torch.tensor(train["exposure"].values, dtype=torch.float32)
exposure_test = torch.tensor(test["exposure"].values, dtype=torch.float32)

y_train = torch.tensor(train["deaths"].values, dtype=torch.float32)
y_test = torch.tensor(test["deaths"].values, dtype=torch.float32)

glm = nn.Linear(X_train.shape[1], 1, bias=True)
mlp = MLP(21, 64, 32)

nn.init.zeros_(mlp.output.weight)
nn.init.zeros_(mlp.output.bias)

optimizer = torch.optim.Adam(list(glm.parameters()) + list(mlp.parameters()), lr=1e-3)

epochs = 200

for epoch in range(epochs):
    glm.train()
    mlp.train()

    optimizer.zero_grad()

    glm_effect = glm(X_train).squeeze(1)
    temp_effect = mlp(X_temp_train)

    log_mu = torch.log(exposure_train) + glm_effect + temp_effect

    loss = nn.functional.poisson_nll_loss(log_mu, y_train, log_input=True)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(glm.parameters()) + list(mlp.parameters()), 1.0)
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f}")

glm.eval()
mlp.eval()

with torch.no_grad():
    log_mu_test = (torch.log(exposure_test) + glm(X_test).squeeze(1) + mlp(X_temp_test))
    pred_test = torch.exp(log_mu_test)

test["prediction"] = pred_test.numpy()

weekly_pred = test.groupby("week", as_index=False).agg(observed=("deaths", "sum"), predicted=("prediction", "sum"))

mae = mean_absolute_error(weekly_pred["observed"], weekly_pred["predicted"])
rmse = np.sqrt(mean_squared_error(weekly_pred["observed"], weekly_pred["predicted"]))

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

for year in [2023, 2024]:
    d = weekly_pred[weekly_pred["week"].dt.year == year]

    plt.figure(figsize=(14, 6))
    plt.plot(d["week"], d["observed"], label="Observed", linewidth=2)
    plt.plot(d["week"], d["predicted"], label="GLM + MLP")
    plt.title(f"Weekly mortality — {year}")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

class MLP(nn.Module):
    def __init__(self, d_input, d_hidden_1, d_hidden_2, d_output):
        super().__init__()
        self.layer1 = nn.Linear(d_input, d_hidden_1)
        self.layer2 = nn.Linear(d_hidden_1, d_hidden_2)
        self.output = nn.Linear(d_hidden_2, d_output)

    def forward(self, x):
        x1 = nn.functional.gelu(self.layer1(x))
        x2 = nn.functional.gelu(self.layer2(x1))
        return self.output(x2)


data = pd.read_parquet("Data/data_nn.parquet")
data["year"] = data["week"].dt.year

train = data[data["week"].between("2004-01-01", "2022-12-31")].copy()
test = data[data["week"] >= "2023-01-01"].copy()

week_num_train = train["week"].dt.isocalendar().week.astype(float).to_numpy()
week_num_test = test["week"].dt.isocalendar().week.astype(float).to_numpy()

train["week_sin"] = np.sin(2 * np.pi * week_num_train / 52)
train["week_cos"] = np.cos(2 * np.pi * week_num_train / 52)
test["week_sin"] = np.sin(2 * np.pi * week_num_test / 52)
test["week_cos"] = np.cos(2 * np.pi * week_num_test / 52)

temp_cols = [f"{v}_{d}" for v in ["TN", "TM", "TX"] for d in range(1, 8)]

X_train = pd.concat([
    pd.get_dummies(train[["sex", "age_group"]], drop_first=True, dtype=float),
    train[["departement", "year", "week_sin", "week_cos"]].astype(float)
], axis=1)

X_test = pd.concat([
    pd.get_dummies(test[["sex", "age_group"]], drop_first=True, dtype=float),
    test[["departement", "year", "week_sin", "week_cos"]].astype(float)
], axis=1)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = sm.add_constant(X_train, has_constant="add")
X_test = sm.add_constant(X_test, has_constant="add")

glm = sm.GLM(train["deaths"], X_train, family=sm.families.Poisson(), offset=np.log(train["exposure"])).fit()

glm_pred_train = glm.predict(X_train, offset=np.log(train["exposure"]))
glm_pred_test = glm.predict(X_test, offset=np.log(test["exposure"]))

X_temp_train = torch.tensor(train[temp_cols].values, dtype=torch.float32)
X_temp_test = torch.tensor(test[temp_cols].values, dtype=torch.float32)

# Normalize using training data only
mean = X_temp_train.mean(dim=0)
std = X_temp_train.std(dim=0)

X_temp_train = (X_temp_train - mean) / std
X_temp_test = (X_temp_test - mean) / std

y_train = torch.tensor(train["deaths"].values, dtype=torch.float32)
y_test = torch.tensor(test["deaths"].values, dtype=torch.float32)

glm_pred_train = torch.tensor(glm_pred_train.values, dtype=torch.float32)
glm_pred_test = torch.tensor(glm_pred_test.values, dtype=torch.float32)

mlp = MLP(21, 64, 32, 1)

optimizer = torch.optim.Adam(list(mlp.parameters()), lr=1e-3)
criterion = nn.MSELoss()

epochs = 200

for epoch in range(epochs):
    mlp.train()

    optimizer.zero_grad()

    z = mlp(X_temp_train)
    effect = z.squeeze(1)

    pred = glm_pred_train * torch.exp(effect)

    loss = criterion(pred, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f}")

mlp.eval()

with torch.no_grad():
    z_test = mlp(X_temp_test).squeeze(1)
    pred_test = glm_pred_test * torch.exp(z_test)

pred_test = pred_test.numpy()
test["prediction"] = pred_test

new_results = test[["week", "deaths", "prediction"]].groupby("week", as_index=False).sum()

mae = np.mean(np.abs(new_results["deaths"] - new_results["prediction"]))
rmse = np.sqrt(np.mean((new_results["deaths"] - new_results["prediction"]) ** 2))

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

weekly_pred = test.groupby("week", as_index=False).agg(observed=("deaths", "sum"), predicted=("prediction", "sum"))
weekly_pred["week"] = pd.to_datetime(weekly_pred["week"])

for year in [2023, 2024]:
    d = weekly_pred[weekly_pred["week"].dt.year == year]

    plt.figure(figsize=(14, 6))
    plt.plot(d["week"], d["observed"], label="Observed", linewidth=2)
    plt.plot(d["week"], d["predicted"], label="MLP + GLM")
    plt.title(f"Weekly mortality — {year}")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
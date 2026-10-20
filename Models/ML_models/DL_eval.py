import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DL_model import Model, MortalityDataset, NonePoissonTokenizer, NonePoissonBaseTokenizer
from torch.utils.data import DataLoader

#Hyperparameters

n_layers = 4
d_embed = 64
n_heads = 8
d_ff = 256
act_fun = nn.GELU()
d_out = 1
batch_size = 128

# Data
print("Loading data ...")
data = pd.read_parquet("Data/data_weekly.parquet")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Splitting data ...")
train = data[data["year"] <= 2019].copy()
valid = data[data["year"].between(2021, 2022)].copy()
test  = data[data["year"].between(2023, 2024)].copy()

train_dataset = MortalityDataset(train)
valid_dataset = MortalityDataset(valid)
test_dataset = MortalityDataset(test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Models

basemodel = Model(n_layers, d_embed, n_heads, d_ff, act_fun, d_out, NonePoissonBaseTokenizer).to(device)
basemodel.load_state_dict(torch.load("Results/Python_results/transformer_weekly_base.pt", weights_only=True))
basemodel.to(device)
basemodel.eval()

tempmodel = Model(n_layers, d_embed, n_heads, d_ff, act_fun, d_out, NonePoissonTokenizer).to(device)
tempmodel.load_state_dict(torch.load("Results/Python_results/transformer_weekly_temp.pt", weights_only=True))
tempmodel.to(device)
tempmodel.eval()

# Metrics

print("Evaluating model ...")
def poisson_deviance(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
    dev = np.zeros_like(y)
    positive = y > 0
    dev[positive] = 2 * (y[positive] * np.log(y[positive] / mu[positive]) - (y[positive] - mu[positive]))
    dev[~positive] = 2 * mu[~positive]
    return dev.sum()

pred_base = []
pred_temp = []
observations = []

with torch.no_grad():
    for batch in test_loader:
        xt, xtemp, sex, age, dep, exposure, deaths = [x.to(device) for x in batch]

        base = basemodel(xt, sex, age, dep, exposure).squeeze(-1)
        temp = tempmodel(xt, xtemp, sex, age, dep, exposure).squeeze(-1)

        pred_base.append(base.cpu())
        pred_temp.append(temp.cpu())
        observations.append(deaths.cpu())

pred_base = torch.cat(pred_base).numpy()
pred_temp = torch.cat(pred_temp).numpy()
observations = torch.cat(observations).numpy()

dates = test_dataset.data["date"].to_numpy()
exposure = test_dataset.data["exposure"].to_numpy()

results = pd.DataFrame({
    "date": dates,
    "observed": observations,
    "base": pred_base,
    "temperature": pred_temp,
    "exposure" : exposure})

print("Observed deaths:", observations.sum())
print("Base predicted deaths:", pred_base.sum())
print("Temperature predicted deaths:", pred_temp.sum())

observed_rate = observations.sum() / valid["exposure"].sum()
base_rate = pred_base.sum() / valid["exposure"].sum()
temp_rate = pred_temp.sum() / valid["exposure"].sum()

print(f"Observed rate: {observed_rate:.8f}")
print(f"Base rate: {base_rate:.8f}")
print(f"Temperature rate: {temp_rate:.8f}")
print("Training rate:", train["deaths"].sum() / train["exposure"].sum())


for year in [2023, 2024]:
    year_data = results[results["date"].dt.year == year]
    daily = year_data.groupby("date")[["observed", "base", "temperature"]].sum().reset_index()

    rmse_base = np.sqrt(np.mean((daily["observed"] - daily["base"]) ** 2))
    rmse_temp = np.sqrt(np.mean((daily["observed"] - daily["temperature"]) ** 2))
    dev_base = poisson_deviance(daily["observed"], daily["base"])
    dev_temp = poisson_deviance(daily["observed"], daily["temperature"])

    print(f"\n{year}")
    print(f"Observed: {year_data['observed'].sum():.0f}")
    print(f"Base predicted: {year_data['base'].sum():.0f}")
    print(f"Temperature predicted: {year_data['temperature'].sum():.0f}")
    print(f"Base ratio: {year_data['base'].sum() / year_data['observed'].sum():.3f}")
    print(f"Temperature ratio: {year_data['temperature'].sum() / year_data['observed'].sum():.3f}")
    print(f"Base RMSE: {rmse_base:.4f}")
    print(f"Temperature RMSE: {rmse_temp:.4f}")
    print(f"Base Poisson deviance: {dev_base:.2f}")
    print(f"Temperature Poisson deviance: {dev_temp:.2f}")

    x = results[results["date"].dt.year == year]
    exposure = x["exposure"].sum()
    print(f"Observed rate: {x['observed'].sum() / exposure:.8f}")
    print(f"Base rate: {x['base'].sum() / exposure:.8f}")
    print(f"Temperature rate: {x['temperature'].sum() / exposure:.8f}")

    plt.figure(figsize=(14, 6))
    plt.plot(daily["date"], daily["observed"], label="Actual")
    plt.plot(daily["date"], daily["base"], label="Base")
    plt.plot(daily["date"], daily["temperature"], label="Temperature")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    plt.title(f"Transformer predictions vs actual deaths — {year}")
    plt.legend()
    plt.tight_layout()
    plt.show()
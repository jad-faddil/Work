import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from DL_model import Model, MortalityDataset, NonePoissonTokenizer, NonePoissonBaseTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_layers = 4
d_embed = 64
n_heads = 8
d_ff = 256
act_fun = nn.GELU()
d_out = 1
batch_size = 128

n_steps = 10000
lr = 0.001

# Data
print("Loading data ...")
data = pd.read_parquet("Data/data_weekly.parquet")

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

# Model
model = Model(n_layers, d_embed, n_heads, d_ff, act_fun, d_out, NonePoissonTokenizer).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = F.mse_loss

# Training loop

best_loss = torch.inf
patience = 5
counter = 0
n_epoch = 0

train_iter = iter(train_loader)

print("Starting training ...")
for step in range(n_steps):

    model.train()
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)
        n_epoch += 1
        print(f"{n_epoch} epochs completed")

    xt, xtemp, sex, age, dep, exposure, deaths = [x.to(device) for x in batch]
    optimizer.zero_grad()
    pred = model(xt, xtemp, sex, age, dep, exposure)
    loss = loss_fn(pred.squeeze(-1), deaths)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()

    if (step + 1) % 1000 == 0:
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                xt, xtemp, sex, age, dep, exposure, deaths = [x.to(device) for x in batch]
                pred = model(xt, xtemp, sex, age, dep, exposure)
                valid_loss += loss_fn(pred.squeeze(-1), deaths).item()

        valid_loss /= len(valid_loader)

        print(f"Step {step+1}: train = {train_loss:.4f}, valid = {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model.state_dict(), "Results/Python_results/transformer_weekly_temp.pt")
        else:
            counter += 1
            if counter >= patience:
                break
    elif step % 100 == 0:
        print(f"Step {step}: train = {train_loss:.4f}")

print("Training completed. Model Saved")
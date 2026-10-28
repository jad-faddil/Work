import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# Load
data = pd.read_parquet("Data/data_weekly.parquet")
data["year"] = data["week"].dt.year

for col in ["sex", "age_group", "departement"]:
    data[col] = data[col].astype("category")
    
# Split
train = data[data["year"] <= 2020]
valid = data[data["year"].between(2021, 2022)]
test = data[data["year"].between(2023, 2024)]

# Features

features_base = ["sex", "age_group", "departement", "year", "exposure"]
features_temp = features_base + ["TN_min", "TN_max", "TX_min", "TX_max", "TN_mean", "TX_mean"]

X_train_base, X_valid_base, X_test_base = train[features_base], valid[features_base], test[features_base]
X_train_temp, X_valid_temp, X_test_temp = train[features_temp], valid[features_temp], test[features_temp]

y_train = train["deaths"]
y_valid = valid["deaths"]

# XGBoost parameters
params = dict(objective="count:poisson", n_estimators=1000, learning_rate=0.05, 
              max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
              enable_categorical=True, tree_method="hist", eval_metric="poisson-nloglik", 
              early_stopping_rounds=50, n_jobs=-1)

# Baseline model
model_base = xgb.XGBRegressor(**params)
model_base.fit(X_train_base, y_train, eval_set=[(X_valid_base, y_valid)], verbose=50)

# Temperature model
model_temp = xgb.XGBRegressor(**params)
model_temp.fit(X_train_temp, y_train, eval_set=[(X_valid_temp, y_valid)], verbose=50)

model_base.save_model("Results/Python_results/model_weekly_base.json")
model_temp.save_model("Results/Python_results/model_weekly_temp.json")
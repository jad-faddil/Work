import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class Attention(nn.Module):
    def __init__(self, d_embed : int, d_hidden : int):
        super().__init__()
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.key = nn.Linear(d_embed, d_hidden, bias = False)
        self.query = nn.Linear(d_embed, d_hidden, bias = False)
        self.value = nn.Linear(d_embed, d_hidden, bias= False)

    def forward(self, x : torch.Tensor):
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        A = nn.Softmax(dim=-1)((Q @ K.transpose(-2, -1)) / np.sqrt(self.d_hidden))
        return A @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_embed : int, n_heads : int):
            super().__init__()
            self.d_embed = d_embed
            self.d_hidden = d_embed // n_heads
            self.heads = nn.ModuleList([Attention(self.d_embed, self.d_hidden) for _ in range(n_heads)])
            self.output = nn.Linear(self.d_embed, self.d_embed)

    def forward(self, x : torch.Tensor):
        heads_forward = [head(x) for head in self.heads]
        concat = torch.cat(heads_forward, dim=-1)
        return self.output(concat)

class FeedForward(nn.Module):
    def __init__(self, d_embed : int, d_ff : int, act_fun : nn.Module):
        super().__init__()
        self.d_embed = d_embed
        self.d_ff = d_ff
        self.W1 = nn.Linear(self.d_embed, self.d_ff)
        self.W2 = nn.Linear(self.d_ff, self.d_embed)
        self.act_fun = act_fun

    def forward(self, x : torch.Tensor):
        return self.W2(self.act_fun(self.W1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_embed : int, n_heads : int, d_ff : int, act_fun : nn.Module):
            super().__init__()
            self.mha = MultiHeadAttention(d_embed, n_heads)
            self.ff = FeedForward(d_embed, d_ff, act_fun)
            self.norm1 = nn.LayerNorm(d_embed)
            self.norm2 = nn.LayerNorm(d_embed)

    def forward(self, x : torch.Tensor):
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, n_layers : int, d_embed : int, n_heads : int, d_ff : int, act_fun : nn.Module, d_out : int):
            super().__init__()
            self.n_layers = n_layers
            self.d_embed = d_embed
            self.n_heads = n_heads
            self.d_ff = d_ff
            self.act_fun = act_fun
            self.d_out = d_out
            self.layers = nn.ModuleList([TransformerBlock(self.d_embed, self.n_heads, self.d_ff, self.act_fun) for _ in range(self.n_layers)])
            self.out =  nn.Linear(self.d_embed, self.d_out)

    def forward(self, x : torch.Tensor):
        for layer in self.layers: 
            x = layer(x)
        x = x[:, 0, :]    
        return self.out(x)

class TempTokenizer(nn.Module):
    def __init__(self, d_embed : int):
        super().__init__()
        self.d_embed = d_embed
        self.temporal = nn.Linear(3, self.d_embed)
        self.temperature = nn.Linear(6, self.d_embed)
        self.sex = nn.Embedding(2, self.d_embed)
        self.age = nn.Embedding(9, self.d_embed)
        self.departement = nn.Embedding(95, self.d_embed)
        self.pred_token = nn.Parameter(torch.randn(1, 1, d_embed))

    def forward(self, x_temporal, x_temperature, x_sex, x_age, x_departement):
        temporal = self.temporal(x_temporal)
        temperature = self.temperature(x_temperature)
        demographic = self.sex(x_sex) + self.age(x_age) + self.departement(x_departement)
        pred_token = self.pred_token.expand(x_temporal.size(0), -1, -1)
        return torch.cat([pred_token, temporal.unsqueeze(1), temperature.unsqueeze(1), demographic.unsqueeze(1)], dim=1)

class BaseTokenizer(nn.Module):
    def __init__(self, d_embed: int):
        super().__init__()
        self.d_embed = d_embed
        self.temporal = nn.Linear(3, self.d_embed)
        self.sex = nn.Embedding(2, self.d_embed)
        self.age = nn.Embedding(9, self.d_embed)
        self.departement = nn.Embedding(95, self.d_embed)
        self.pred_token = nn.Parameter(torch.randn(1, 1, d_embed))

    def forward(self, x_temporal, x_sex, x_age, x_departement):
        temporal = self.temporal(x_temporal)
        demographic = self.sex(x_sex) + self.age(x_age) + self.departement(x_departement)
        pred_token = self.pred_token.expand(x_temporal.size(0), -1, -1)

        return torch.cat([pred_token,temporal.unsqueeze(1),demographic.unsqueeze(1)], dim=1)
    
class Model(nn.Module):
    def __init__(self, n_layers : int, d_embed : int, n_heads : int, d_ff : int, act_fun : nn.Module, d_out : int, Tokenizer : nn.Module):
        super().__init__()
        self.n_layers = n_layers
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.act_fun = act_fun
        self.d_out = d_out
        self.tokenizer = Tokenizer(d_embed)
        self.architecture = Transformer(n_layers, d_embed, n_heads, d_ff, act_fun, d_out)

    def forward(self, *args):
        x = self.tokenizer(*args)
        return self.architecture(x)

class MortalityDataset(Dataset):
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.data["week"] = self.data["date"].dt.isocalendar().week.astype(float)
        self.temporal = torch.tensor(self.data[["year", "week"]].values, dtype=torch.float32)
        self.temperature = torch.tensor(self.data[["TN_min", "TN_min", "TX_max", "TX_max"]].values, dtype=torch.float32)
        self.sex = torch.tensor(self.data["sex"].map({"F": 0, "M": 1}).values, dtype=torch.long)
        self.age = torch.tensor(self.data["age_group"].map({age: i for i, age in enumerate(["20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64"])}).values, dtype=torch.long)
        self.departement = torch.tensor(self.data["departement"].astype(int).values - 1, dtype=torch.long)
        self.deaths = torch.tensor(self.data["deaths"].values.astype(np.float32))
        self.exposure = torch.tensor(self.data["exposure"].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.temporal[idx], self.temperature[idx], self.sex[idx], self.age[idx], self.departement[idx], self.exposure[idx], self.deaths[idx]

class NonePoissonTokenizer(nn.Module):
    def __init__(self, d_embed : int):
        super().__init__()
        self.d_embed = d_embed
        self.temporal = nn.Linear(2, self.d_embed)
        self.temperature = nn.Linear(4, self.d_embed)
        self.sex = nn.Embedding(2, self.d_embed)
        self.age = nn.Embedding(9, self.d_embed)
        self.departement = nn.Embedding(95, self.d_embed)
        self.exposure = nn.Linear(1, self.d_embed)
        self.pred_token = nn.Parameter(torch.randn(1, 1, d_embed))

    def forward(self, x_temporal, x_temperature, x_sex, x_age, x_departement, x_exposure):
        temporal = self.temporal(x_temporal)
        temperature = self.temperature(x_temperature)
        demographic = self.sex(x_sex) + self.age(x_age) + self.departement(x_departement)
        pred_token = self.pred_token.expand(x_temporal.size(0), -1, -1)
        x_exposure = torch.log(x_exposure).unsqueeze(-1)
        exposure = self.exposure(x_exposure)
        return torch.cat([pred_token, temporal.unsqueeze(1), temperature.unsqueeze(1), demographic.unsqueeze(1), exposure.unsqueeze(1)], dim=1)

class NonePoissonBaseTokenizer(nn.Module):
    def __init__(self, d_embed : int):
        super().__init__()
        self.d_embed = d_embed
        self.temporal = nn.Linear(2, self.d_embed)
        self.sex = nn.Embedding(2, self.d_embed)
        self.age = nn.Embedding(9, self.d_embed)
        self.departement = nn.Embedding(95, self.d_embed)
        self.exposure = nn.Linear(1, self.d_embed)
        self.pred_token = nn.Parameter(torch.randn(1, 1, d_embed))

    def forward(self, x_temporal, x_sex, x_age, x_departement, x_exposure):
        temporal = self.temporal(x_temporal)
        demographic = self.sex(x_sex) + self.age(x_age) + self.departement(x_departement)
        pred_token = self.pred_token.expand(x_temporal.size(0), -1, -1)
        x_exposure = torch.log(x_exposure).unsqueeze(-1)
        exposure = self.exposure(x_exposure)
        return torch.cat([pred_token, temporal.unsqueeze(1), demographic.unsqueeze(1), exposure.unsqueeze(1)], dim=1)
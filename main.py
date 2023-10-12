import pandas as pd

## Read Data

data = pd.read_csv("Sonar.csv")

X = data.iloc[:, 0:60]  # Features
y = data.iloc[:, 60]  # Variavel alvo

# print(y)

## Binary encoding of labels
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)  # códificando os rótulos da variavel alvo. De M | R PARA 0 | 1

# print(encoder.classes_)

# print(y)

import torch

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

## Creating a Model

##
# The more parameters a model has, heuristically we believe that it is more powerful.
# Should you use a model with fewer layers but more parameters on each layer,
# or a model with more layers but less parameters each?
# #

## A model with more parameters on each layer is called a wider model.

import torch.nn as nn


class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(60, 180)
        self.relu = nn.ReLU()
        self.output = nn.Linear(180, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


## A model with more layer is called a deeper model


class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

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

# Creating a Model

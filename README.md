# Neural-network
# Antibody Liability Predictor

This repository contains a PyTorch neural network for predicting antibody
developability liabilities from ESM-2 embeddings.

## Model architecture
- Input: 640-D (VH 320 + VL 320)
- Output: 4 regression values
- Architecture: MLP (128 → 64)

## Usage
```python
import torch
from model import LiabilityPredictor

model = LiabilityPredictor(input_dim=640)
model.load_state_dict(torch.load("liability_predictor.pt", map_location="cpu"))
model.eval()

# x should be shape (640,) or (batch, 640)
y_pred = model(x)

import torch
import torch.nn as nn
from .dkl_model import DKLModel


class QuantileDKL(nn.Module):
    def __init__(self, input_dim, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        self.base = DKLModel(input_dim)
        self.head = nn.Linear(1, len(quantiles))

    def forward(self, x):
        mvn = self.base(x)
        pred = self.head(mvn.mean.unsqueeze(-1))
        return pred

    def to(self, device):
        self.base.to(device)
        self.head.to(device)
        return self

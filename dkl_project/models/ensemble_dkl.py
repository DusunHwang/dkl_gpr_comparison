import torch
import torch.nn as nn
from .dkl_model import DKLModel


class EnsembleDKL(nn.Module):
    def __init__(self, input_dim, num_models: int = 5):
        super().__init__()
        self.models = nn.ModuleList([DKLModel(input_dim) for _ in range(num_models)])

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        means = torch.stack([o.mean for o in outputs], dim=0)
        variances = torch.stack([o.variance for o in outputs], dim=0)
        mean = means.mean(0)
        total_var = variances.mean(0) + means.var(0)
        return mean, total_var.sqrt()

    def to(self, device):
        for m in self.models:
            m.to(device)
        return self

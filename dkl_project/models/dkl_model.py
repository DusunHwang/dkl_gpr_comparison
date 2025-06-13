import torch
import torch.nn as nn
import gpytorch


class FeatureExtractor(nn.Sequential):
    def __init__(self, input_dim):
        layers = [
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ]
        super().__init__(*layers)
        self.out_dim = 64


class GPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, feature_dim):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(128)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            torch.randn(128, feature_dim),
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, features):
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim)
        self.gp_model = GPRegressionModel(self.feature_extractor.out_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp_model(features)

    def to(self, device):
        super().to(device)
        self.feature_extractor.to(device)
        self.gp_model.to(device)
        return self

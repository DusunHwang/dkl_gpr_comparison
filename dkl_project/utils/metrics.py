import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def coverage_within(y_true, y_pred_mean, y_sigma, z=1.96):
    lower = y_pred_mean - z * y_sigma
    upper = y_pred_mean + z * y_sigma
    inside = ((y_true >= lower) & (y_true <= upper)).float().mean().item()
    return inside


def sigma_mae_corr(y_sigma, errors):
    y_sigma = y_sigma.detach().cpu().numpy().ravel()
    errors = errors.detach().cpu().numpy().ravel()
    return np.corrcoef(y_sigma, np.abs(errors))[0, 1]

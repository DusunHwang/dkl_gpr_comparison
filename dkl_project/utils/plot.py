import matplotlib.pyplot as plt
import torch


def plot_pred_vs_true(y_true, y_pred, r2, mse, path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(f'R2={r2:.3f}, MSE={mse:.3f}')
    plt.savefig(path)
    plt.close()


def plot_sigma_mae(sigmas, errors, path):
    plt.figure()
    plt.scatter(sigmas, errors, alpha=0.5)
    plt.xlabel('Sigma')
    plt.ylabel('Absolute Error')
    plt.savefig(path)
    plt.close()

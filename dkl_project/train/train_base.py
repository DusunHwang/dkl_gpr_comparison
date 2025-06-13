import argparse
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import gpytorch
from pathlib import Path

from data.preprocess import preprocess
from models.dkl_model import DKLModel
from utils.metrics import regression_metrics, coverage_within, sigma_mae_corr
from utils.plot import plot_pred_vs_true, plot_sigma_mae


def train(args):
    df = pd.read_csv(args.csv_path)
    x_cols = args.x_columns.split(',')
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(df, x_cols, args.target_column)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DKLModel(X_train.shape[1]).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_model, num_data=y_train.size(0))

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    best_loss = float('inf')
    patience = 10
    counter = 0
    log = []
    for epoch in range(args.epochs):
        model.train()
        likelihood.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.squeeze(-1).to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        scheduler.step()
        epoch_loss /= len(train_loader.dataset)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_loss = -mll(val_out, y_val.squeeze(-1).to(device)).item()
        log.append({'epoch': epoch, 'train_loss': epoch_loss, 'val_loss': val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_base.pt')
        else:
            counter += 1
            if counter >= patience:
                break
    # Evaluation
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        test_out = model(X_test.to(device))
        preds = test_out.mean.cpu()
        sigma = test_out.variance.sqrt().cpu()
        y_true = y_test.squeeze()
        mae, rmse, r2 = regression_metrics(y_true, preds)
        cover = coverage_within(y_true, preds, sigma)
        corr = sigma_mae_corr(sigma, preds - y_true)

    Path('results').mkdir(exist_ok=True)
    plot_pred_vs_true(y_true.numpy(), preds.numpy(), r2, rmse**2, 'results/pred_vs_true.png')
    plot_sigma_mae(sigma.numpy(), torch.abs(preds - y_true).numpy(), 'results/sigma_mae.png')

    import csv
    Path('logs').mkdir(exist_ok=True)
    with open('logs/train_log.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch','train_loss','val_loss'])
        writer.writeheader()
        writer.writerows(log)

    Path('reports').mkdir(exist_ok=True)
    with open('reports/eval.md', 'w') as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\nCoverage: {cover:.4f}\nSigma_MAE_corr: {corr:.4f}\n")
        if r2 > 0.85:
            f.write('\nhigh_performance: true\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='data/example_dataset.csv')
    parser.add_argument('--x_columns', required=True)
    parser.add_argument('--target_column', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()

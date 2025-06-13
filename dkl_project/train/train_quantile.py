import argparse
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import gpytorch
from pathlib import Path

from data.preprocess import preprocess
from models.quantile_dkl import QuantileDKL
from utils.loss import pinball_loss
from utils.plot import plot_pred_vs_true


def train(args):
    df = pd.read_csv(args.csv_path)
    x_cols = args.x_columns.split(',')
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(df, x_cols, args.target_column)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QuantileDKL(X_train.shape[1]).to(device)

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
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = pinball_loss(preds, yb, model.quantiles)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        scheduler.step()
        epoch_loss /= len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device))
            val_loss = pinball_loss(val_preds, y_val.to(device), model.quantiles).item()
        log.append({'epoch': epoch, 'train_loss': epoch_loss, 'val_loss': val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_quantile.pt')
        else:
            counter += 1
            if counter >= patience:
                break

    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu()
    median = preds[:, 1]
    r2 = 0
    mse = torch.mean((median - y_test.squeeze()) ** 2).item()
    Path('results').mkdir(exist_ok=True)
    plot_pred_vs_true(y_test.numpy(), median.numpy(), r2, mse, 'results/pred_vs_true_quantile.png')

    import csv
    Path('logs').mkdir(exist_ok=True)
    with open('logs/train_log.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch','train_loss','val_loss'])
        writer.writeheader()
        writer.writerows(log)

    Path('reports').mkdir(exist_ok=True)
    with open('reports/eval.md', 'w') as f:
        f.write(f"MSE: {mse:.4f}\n")


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

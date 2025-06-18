import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def generate_data(n_samples=50000, n_features=20):
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df['y'] = y
    return df


def split_and_save(df):
    os.makedirs('data', exist_ok=True)
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)
    train.to_csv('data/train.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    return train, val, test


def preprocess(train, val, test, feature_cols, target_col):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(train[feature_cols])
    y_train = scaler_y.fit_transform(train[[target_col]])
    X_val = scaler_x.transform(val[feature_cols])
    y_val = scaler_y.transform(val[[target_col]])
    X_test = scaler_x.transform(test[feature_cols])
    y_test = scaler_y.transform(test[[target_col]])
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.seq(x)


def train_model(device, X_train, y_train, X_val, y_val, epochs=200):
    model = Net(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    log = []
    start = time.perf_counter()
    for epoch in tqdm(range(epochs), desc=f'Training ({device})'):
        model.train()
        total = 0
        for xb, yb in tqdm(train_loader, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, y_val.to(device)).item()
        log.append({'epoch': epoch, 'train_loss': train_loss,
                    'val_loss': val_loss, 'device': device})
    runtime = time.perf_counter() - start
    with torch.no_grad():
        val_pred = model(X_val.to(device))
        val_true = y_val
        val_r2 = r2_score(val_true.cpu().numpy(), val_pred.cpu().numpy())
    return model, log, runtime, val_r2


def evaluate(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
    y_true = y_test.cpu().numpy()
    mse = mean_squared_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    return preds, mse, r2


def plot_results(y_true, y_pred, r2, mse, path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(f'R2={r2:.3f}, MSE={mse:.3f}')
    plt.savefig(path)
    plt.close()


def main(samples=50000, epochs=200):
    if all(os.path.exists(f"data/{name}.csv") for name in ["train", "val", "test"]):
        train_df = pd.read_csv("data/train.csv")
        val_df = pd.read_csv("data/val.csv")
        test_df = pd.read_csv("data/test.csv")
    else:
        df = generate_data(n_samples=samples)
        train_df, val_df, test_df = split_and_save(df)
    feature_cols = [c for c in df.columns if c != 'y']
    target_col = 'y'
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(
        train_df, val_df, test_df, feature_cols, target_col)

    logs = []
    cpu_model, log_cpu, cpu_time, cpu_r2 = train_model('cpu', X_train, y_train,
                                                       X_val, y_val,
                                                       epochs=epochs)
    logs.extend(log_cpu)
    best_model = cpu_model
    best_r2 = cpu_r2
    gpu_time = None
    if torch.cuda.is_available():
        gpu_model, log_gpu, gpu_time, gpu_r2 = train_model('cuda', X_train, y_train,
                                                           X_val, y_val,
                                                           epochs=epochs)
        logs.extend(log_gpu)
        if gpu_r2 > best_r2:
            best_model = gpu_model
            best_r2 = gpu_r2
    os.makedirs('logs', exist_ok=True)
    pd.DataFrame(logs).to_csv('logs/train_log.csv', index=False)

    preds, mse, r2 = evaluate(best_model, X_test, y_test, 'cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)
    plot_results(y_test.numpy(), preds, r2, mse, 'plots/train_vs_test.png')
    os.makedirs('models', exist_ok=True)
    torch.save(best_model.state_dict(), 'models/final_model.pt')

    os.makedirs('reports', exist_ok=True)
    with open('reports/eval.md', 'w') as f:
        f.write(f'MSE: {mse:.4f}\nR2: {r2:.4f}\n')
        if gpu_time is not None:
            f.write(f'CPU_time: {cpu_time:.2f}s\nGPU_time: {gpu_time:.2f}s\n')
        else:
            f.write(f'CPU_time: {cpu_time:.2f}s\nGPU_time: N/A\n')
        if best_r2 > 0.85:
            f.write('high_performance: true\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare CPU vs GPU training')
    parser.add_argument('--samples', type=int, default=50000,
                        help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs for each device')
    args = parser.parse_args()

    main(samples=args.samples, epochs=args.epochs)

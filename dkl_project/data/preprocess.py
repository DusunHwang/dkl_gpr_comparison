import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch


def preprocess(df: pd.DataFrame, x_columns, target_column):
    """Preprocess dataframe with mixed types."""
    x_columns = list(x_columns)
    # Identify categorical columns
    cat_cols = [c for c in x_columns if str(df[c].dtype) in ['object', 'category']]
    num_cols = [c for c in x_columns if c not in cat_cols]

    df_num = df[num_cols].copy()
    df_cat = pd.get_dummies(df[cat_cols], drop_first=False)
    X = pd.concat([df_num, df_cat], axis=1)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(df[[target_column]])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42
    )  # 0.1765 * 0.85 ~ 0.15

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    return (
        to_tensor(X_train),
        to_tensor(y_train),
        to_tensor(X_val),
        to_tensor(y_val),
        to_tensor(X_test),
        to_tensor(y_test),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path")
    parser.add_argument("--x_columns")
    parser.add_argument("--target_column")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    x_cols = args.x_columns.split(',')
    preprocess(df, x_cols, args.target_column)

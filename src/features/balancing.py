from __future__ import annotations
from typing import Tuple
import pandas as pd


def balance_undersample(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Undersample each class to the size of the smallest class."""
    counts = df[target_col].value_counts()
    if len(counts) <= 1:
        return df.copy()
    min_count = int(counts.min())
    parts = [
        df[df[target_col] == cls].sample(n=min_count, replace=False)
        for cls in counts.index
    ]
    return pd.concat(parts).sample(frac=1).reset_index(drop=True)


def balance_features_labels(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    combined = balance_undersample(X.assign(_y=y), "_y")
    return combined.drop(columns="_y"), combined["_y"]


def balance_and_report(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    print("\nClass distribution before balancing (train only):")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  Class {cls}: {count} cases")
    X_train, y_train = balance_features_labels(X_train, y_train)
    print("\nClass distribution after balancing (train only):")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  Class {cls}: {count} cases")
    return X_train, y_train



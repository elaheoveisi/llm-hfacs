from __future__ import annotations
from typing import Tuple
import pandas as pd


def balance_undersample(df: pd.DataFrame, target_col: str, seed: int) -> pd.DataFrame:
    """Undersample each class to the size of the smallest class."""
    counts = df[target_col].value_counts()
    if len(counts) <= 1:
        return df.copy()
    min_count = int(counts.min())
    parts = [df[df[target_col] == cls].sample(n=min_count, replace=False, random_state=seed)
             for cls in counts.index]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def balance_features_labels(
    X: pd.DataFrame, y: pd.Series, seed: int = 7
) -> Tuple[pd.DataFrame, pd.Series]:
    combined = X.assign(_y=y)
    combined_bal = balance_undersample(combined, "_y", seed)
    y_bal = combined_bal.pop("_y")
    return combined_bal, y_bal

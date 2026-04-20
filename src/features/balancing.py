from __future__ import annotations
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

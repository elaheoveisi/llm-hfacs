from __future__ import annotations
from typing import Tuple
import pandas as pd


def balance_undersample(df: pd.DataFrame, target_col: str, seed: int) -> pd.DataFrame:
    """Undersample each class to the size of the smallest class."""
    counts = df[target_col].value_counts()
    if len(counts) <= 1:
        return df.copy()
    min_count = int(counts.min())
    parts = [
        df[df[target_col] == cls].sample(
            n=min_count, replace=False, random_state=seed
        )  # filters the DataFrame to only rows from one class.
        for cls in counts.index
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def balance_features_labels(
    X: pd.DataFrame, y: pd.Series, seed: int = 7
) -> Tuple[pd.DataFrame, pd.Series]:
    combined = balance_undersample(X.assign(_y=y), "_y", seed)
    return combined.drop(columns="_y"), combined["_y"]


"""This function takes your feature table X and label column y,
 temporarily combines them into one DataFrame by adding y as
   a new column called _y, then calls balance_undersample()
     to make each class in _y have the same number of rows by
       randomly removing extra rows from larger classes. 
       After balancing, it separates the data again: 
       it returns the balanced features by dropping _y, 
       and returns the balanced labels from combined["_y"]. 
       The reason it combines them first is to make sure that
         when rows are removed during undersampling,
           the features and labels stay matched correctly."""

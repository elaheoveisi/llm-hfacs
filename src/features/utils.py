from __future__ import annotations
import os
from typing import Dict, List, Tuple
import pandas as pd
import yaml


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataset(config: dict) -> pd.DataFrame:
    return pd.read_csv(config['paths']['processed_csv'])


def get_hfacs_feature_cols(config: dict) -> List[str]:
    _targets = {'Error', 'Violation'}
    return [
        col
        for cat, subcats in config['hfacs_categories'].items()
        if cat not in _targets
        for col in subcats
    ]


def filter_tied_rows(
    df: pd.DataFrame,
    y3: pd.Series,
) -> Tuple:
    dropped = (y3 == -1).sum()
    if dropped:
        print(f"\nDropped full-tie rows: {dropped}")
    mask = y3 != -1
    return df.loc[mask].copy(), y3.loc[mask]


def print_class_distribution(y3: pd.Series) -> None:
    labels = {0: "Neither", 1: "Error", 2: "Violation"}
    print("\nClass distribution after scoring and grouping:")
    for group, count in y3.value_counts().sort_index().items():
        print(f"  {labels.get(group, str(group))} ({group}): {count}")


def make_three_class_target(
    df: pd.DataFrame,
    error_weights: Dict[str, float],
    viol_weights: Dict[str, float],
) -> pd.Series:
    """Build 3-class target: 0=Neither, 1=Error, 2=Violation, -1=drop (full tie).

    Rules:
    - Only Violation active            → 2
    - Only Error active                → 1
    - Neither active                   → 0
    - Both active, viol_count > err_count → 2
    - Both active, err_count > viol_count → 1
    - Both active, counts equal, viol_wsum > err_wsum → 2
    - Both active, counts equal, err_wsum > viol_wsum → 1
    - Both active, counts equal, weights equal → -1 (drop)
    """
    def _numeric(cols):
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.DataFrame(0, index=df.index, columns=cols[:1] or ["_empty"])
        return df[present].apply(pd.to_numeric, errors="coerce").fillna(0)

    def _wsum(cols_weights):
        total = pd.Series(0.0, index=df.index)
        for c, w in cols_weights.items():
            if c in df.columns:
                total += (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(float) * w
        return total

    err_active = _numeric(list(error_weights)) > 0
    vio_active = _numeric(list(viol_weights)) > 0

    err_flag = err_active.any(axis=1).astype(int)
    vio_flag = vio_active.any(axis=1).astype(int)
    err_count = err_active.sum(axis=1)
    vio_count = vio_active.sum(axis=1)
    err_wsum = _wsum(error_weights)
    vio_wsum = _wsum(viol_weights)

    y3 = pd.Series(-2, index=df.index, dtype=int)

    # Simple cases
    y3[(err_flag == 0) & (vio_flag == 0)] = 0
    y3[(err_flag == 1) & (vio_flag == 0)] = 1
    y3[(err_flag == 0) & (vio_flag == 1)] = 2

    # Both active — resolve by count then weight
    both = (err_flag == 1) & (vio_flag == 1)
    y3[both & (vio_count > err_count)] = 2
    y3[both & (err_count > vio_count)] = 1
    remaining = both & (err_count == vio_count)
    y3[remaining & (vio_wsum > err_wsum)] = 2
    y3[remaining & (err_wsum > vio_wsum)] = 1
    y3[remaining & (err_wsum == vio_wsum)] = -1  # full tie → drop

    return y3


def make_three_class_target_from_config(
    df: pd.DataFrame,
    config: dict,
) -> pd.Series:
    error_weights = config["hfacs_categories"]["Error"]
    viol_weights = config["hfacs_categories"]["Violation"]
    return make_three_class_target(df, error_weights, viol_weights)


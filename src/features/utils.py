from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd


def make_three_class_target(
    df: pd.DataFrame,
    error_col: str,
    viol_col: str,
    error_weights: Dict[str, float],
    viol_weights: Dict[str, float],
    tie_break: str = "error",
    thr: float = 0.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build 3-class target: 0=Neither, 1=Error, 2=Violation.
    Tie-breaking priority: 1) anomaly count, 2) weighted sum, 3) tie_break param."""
    if tie_break not in {"error", "violation"}:
        raise ValueError("tie_break must be 'error' or 'violation'")
    if error_col not in df.columns or viol_col not in df.columns:
        raise KeyError(f"Missing target columns: {error_col} / {viol_col}")

    def _binary_df(weights: Dict[str, float]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in weights:
            if c in df.columns:
                out[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0) > thr).astype(int)
            else:
                out[c] = 0
        return out

    err_flag = (pd.to_numeric(df[error_col], errors="coerce").fillna(0) > thr).astype(int)
    vio_flag = (pd.to_numeric(df[viol_col], errors="coerce").fillna(0) > thr).astype(int)
    err_df, vio_df = _binary_df(error_weights), _binary_df(viol_weights)
    err_count, vio_count = err_df.sum(axis=1), vio_df.sum(axis=1)
    err_wsum = sum(err_df[c] * w for c, w in error_weights.items()) if error_weights else pd.Series(0, index=df.index)
    vio_wsum = sum(vio_df[c] * w for c, w in viol_weights.items()) if viol_weights else pd.Series(0, index=df.index)

    y3 = pd.Series(0, index=df.index, dtype=int)
    y3[(err_flag == 1) & (vio_flag == 0)] = 1
    y3[(err_flag == 0) & (vio_flag == 1)] = 2

    both = (err_flag == 1) & (vio_flag == 1)
    if both.any():
        to_error = both & (err_count > vio_count)
        to_viol = both & (vio_count > err_count)
        tie_count = both & (err_count == vio_count)
        to_error |= tie_count & (err_wsum > vio_wsum)
        to_viol |= tie_count & (vio_wsum > err_wsum)
        tie_weight = tie_count & (err_wsum == vio_wsum)
        if tie_break == "error":
            to_error |= tie_weight
        else:
            to_viol |= tie_weight
        y3[to_error] = 1
        y3[to_viol] = 2

    debug = pd.DataFrame({
        "Error_flag": err_flag,
        "Violation_flag": vio_flag,
        "Err_anom_count": err_count,
        "Viol_anom_count": vio_count,
        "Err_weight_sum": err_wsum,
        "Viol_weight_sum": vio_wsum,
        "y3": y3,
    }, index=df.index)
    return y3, debug

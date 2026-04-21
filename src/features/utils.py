from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd


def make_three_class_target(
    df: pd.DataFrame,
    error_cols: List[str],
    viol_cols: List[str],
    error_weights: Dict[str, float],
    viol_weights: Dict[str, float],
    tie_break: str = "error",
    thr: float = 0.0,
    use_count_tiebreak: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build 3-class target: 0=Neither, 1=Error, 2=Violation.
    A row is flagged as Error/Violation if any of the subcategory columns is active.
    Tie-breaking when both flags are 1:
      use_count_tiebreak=True:  1) count, 2) weighted sum, 3) tie_break param
      use_count_tiebreak=False: 1) weighted sum, 2) tie_break param"""
    if tie_break not in {"error", "violation"}:
        raise ValueError("tie_break must be 'error' or 'violation'")

    def _binary_df(cols: List[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in cols:
            if c in df.columns:
                out[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0) > thr).astype(int)
            else:
                out[c] = 0
        return out

    def _any_flag(cols: List[str]) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(0, index=df.index, dtype=int)
        return df[present].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0) > thr).any(axis=1).astype(int)

    err_flag = _any_flag(error_cols)
    vio_flag = _any_flag(viol_cols)
    err_df = _binary_df(list(error_weights) if error_weights else error_cols)
    vio_df = _binary_df(list(viol_weights) if viol_weights else viol_cols)
    err_count, vio_count = err_df.sum(axis=1), vio_df.sum(axis=1)
    err_wsum = sum(err_df[c] * w for c, w in error_weights.items()) if error_weights else pd.Series(0, index=df.index)
    vio_wsum = sum(vio_df[c] * w for c, w in viol_weights.items()) if viol_weights else pd.Series(0, index=df.index)

    y3 = pd.Series(0, index=df.index, dtype=int)
    y3[(err_flag == 1) & (vio_flag == 0)] = 1
    y3[(err_flag == 0) & (vio_flag == 1)] = 2

    both = (err_flag == 1) & (vio_flag == 1)
    if both.any():
        if use_count_tiebreak:
            to_error = both & (err_count > vio_count)
            to_viol = both & (vio_count > err_count)
            remaining = both & (err_count == vio_count)
        else:
            to_error = pd.Series(False, index=df.index)
            to_viol = pd.Series(False, index=df.index)
            remaining = both
        to_error |= remaining & (err_wsum > vio_wsum)
        to_viol |= remaining & (vio_wsum > err_wsum)
        tie_weight = remaining & (err_wsum == vio_wsum)
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


def make_three_class_target_from_config(
    df: pd.DataFrame,
    config_yaml: dict,
) -> Tuple[pd.Series, pd.DataFrame]:
    hfacs = config_yaml['hfacs_categories']
    error_weights = hfacs['Error']
    viol_weights = hfacs['Violation']
    tie_break = config_yaml.get('svm', {}).get('tie_break', 'error')
    return make_three_class_target(
        df=df,
        error_cols=list(error_weights),
        viol_cols=list(viol_weights),
        error_weights=error_weights,
        viol_weights=viol_weights,
        tie_break=tie_break,
    )


def expand_feature_columns(
    feature_categories: List[str],
    hfacs_categories: Dict[str, List[str]],
) -> List[str]:
    """Expand configured HFACS groups into a de-duplicated feature column list."""
    feature_cols: List[str] = []
    seen = set()
    for cat in feature_categories:
        for col in hfacs_categories.get(cat, []):
            if col not in seen:
                feature_cols.append(col)
                seen.add(col)
    return feature_cols


def coerce_numeric_features(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    """Convert selected feature columns to numeric and validate they exist."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise KeyError(f"Missing feature column: {c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_labels_drop_both(
    df: pd.DataFrame,
    error_col: str,
    viol_col: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build a 3-class target after removing rows flagged as both error and violation."""
    err = (pd.to_numeric(df[error_col], errors="coerce").fillna(0) > 0).astype(int)
    vio = (pd.to_numeric(df[viol_col], errors="coerce").fillna(0) > 0).astype(int)
    both = (err == 1) & (vio == 1)

    y = pd.Series(-1, index=df.index, dtype=int)
    y[(err == 0) & (vio == 0)] = 0
    y[(err == 1) & (vio == 0)] = 1
    y[(err == 0) & (vio == 1)] = 2

    debug = pd.DataFrame({
        "Error_flag": err,
        "Violation_flag": vio,
        "Dropped_both": both.astype(int),
        "y3": y,
    }, index=df.index)
    return y, debug


def build_labels_from_related_columns_drop_both(
    df: pd.DataFrame,
    error_related_cols: List[str],
    violation_related_cols: List[str],
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build labels from any positive value in the underlying error/violation columns.

    Output classes:
    - 0: neither error nor violation columns are active
    - 1: one or more error-related columns active only
    - 2: one or more violation-related columns active only
    - -1: both error-related and violation-related columns active; caller should drop
    """
    err_present = [c for c in error_related_cols if c in df.columns]
    vio_present = [c for c in violation_related_cols if c in df.columns]

    err_df = pd.DataFrame(index=df.index)
    for c in err_present:
        err_df[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(int)

    vio_df = pd.DataFrame(index=df.index)
    for c in vio_present:
        vio_df[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(int)

    err_flag = err_df.any(axis=1).astype(int) if not err_df.empty else pd.Series(0, index=df.index, dtype=int)
    vio_flag = vio_df.any(axis=1).astype(int) if not vio_df.empty else pd.Series(0, index=df.index, dtype=int)
    both = (err_flag == 1) & (vio_flag == 1)

    y = pd.Series(-1, index=df.index, dtype=int)
    y[(err_flag == 0) & (vio_flag == 0)] = 0
    y[(err_flag == 1) & (vio_flag == 0)] = 1
    y[(err_flag == 0) & (vio_flag == 1)] = 2

    debug = pd.DataFrame({
        "Error_flag": err_flag,
        "Violation_flag": vio_flag,
        "Error_related_hits": err_df.sum(axis=1) if not err_df.empty else 0,
        "Violation_related_hits": vio_df.sum(axis=1) if not vio_df.empty else 0,
        "Dropped_both": both.astype(int),
        "y3": y,
    }, index=df.index)
    return y, debug

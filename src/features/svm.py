 
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
  
# --- Load config from YAML ---
import yaml
with open(os.path.join(os.path.dirname(__file__), '../../configs/config.yaml'), 'r', encoding='utf-8') as f:
    config_yaml = yaml.safe_load(f)
 
hfacs_categories = config_yaml['hfacs_categories'] 
svm_cfg = config_yaml['svm']
 
class Config:
    data_file: str = os.path.join('data/processed', svm_cfg.get('data_file', 'step3_hfacs_categories.csv'))
    out_dir: str = './svm_outputs'
    feature_categories: List[str] = svm_cfg['feature_columns']
    # Target columns (Level 1 flags)
    target_columns: List[str] = svm_cfg['target_columns']
    error_target_col: str = 'Error'
    viol_target_col: str = 'Violation'
    test_size: float = 0.20  
    seed: int = svm_cfg.get('seed', 7)
    degree_grid: Tuple[int, ...] = tuple(svm_cfg.get('degree_grid', [2, 3]))
    c_grid: Tuple[float, ...] = tuple(svm_cfg.get('c_grid', [0.01, 0.1, 1, 10, 100]))
    tie_break: str = 'error'
 
CFG = Config()
 
# Weights 
error_weights: Dict[str, float] = config_yaml.get('error_weights', {})
viol_weights: Dict[str, float] = config_yaml.get('viol_weights', {})
ERROR_ANOM_COLS: List[str] = list(error_weights.keys())
VIOL_ANOM_COLS: List[str] = list(viol_weights.keys())
 
 #target building
def _safe_binary_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return a 0/1 dataframe for requested cols; missing cols are treated as 0."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0)
            out[c] = (s > 0).astype(int)
        else:
            out[c] = 0
    return out
 
 
def make_three_class_target(
    df: pd.DataFrame,
    error_col: str,
    viol_col: str,
    tie_break: str = "error",
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build 3-class target: 0=neither, 1=Error, 2=Violation. 
    Tie-breaking: 1) count, 2) weighted sum, 3) remove if unresolved."""
    if tie_break not in {"error", "violation"}:
        raise ValueError("tie_break must be 'error' or 'violation'")
    if error_col not in df.columns or viol_col not in df.columns:
        raise KeyError(f"Missing target columns: {error_col} / {viol_col}")

    err_flag = (pd.to_numeric(df[error_col], errors="coerce").fillna(0) > 0).astype(int)
    vio_flag = (pd.to_numeric(df[viol_col], errors="coerce").fillna(0) > 0).astype(int)
    err_df, vio_df = _safe_binary_df(df, ERROR_ANOM_COLS), _safe_binary_df(df, VIOL_ANOM_COLS)
    err_count, vio_count = err_df.sum(axis=1), vio_df.sum(axis=1)
    err_wsum = sum(err_df[c] * w for c, w in error_weights.items()) if error_weights else pd.Series(0, index=df.index)
    vio_wsum = sum(vio_df[c] * w for c, w in viol_weights.items()) if viol_weights else pd.Series(0, index=df.index)

    y3 = pd.Series(0, index=df.index)
    y3[(err_flag == 1) & (vio_flag == 0)] = 1
    y3[(err_flag == 0) & (vio_flag == 1)] = 2

    both = (err_flag == 1) & (vio_flag == 1)
    if both.any():
        # 1. Use anomaly counts
        to_error = both & (err_count > vio_count)
        to_viol  = both & (vio_count > err_count)
        # 2. If counts are tied, use weighted sum
        tie_count = both & (err_count == vio_count)
        to_error |= tie_count & (err_wsum > vio_wsum)
        to_viol  |= tie_count & (vio_wsum > err_wsum)
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
    
    # Remove ambiguous cases that couldn't be resolved by count or weighted sum
    ambiguous = debug[(debug["Error_flag"] == 1) & (debug["Violation_flag"] == 1) & 
                      (debug["Err_anom_count"] == debug["Viol_anom_count"]) &
                      (debug["Err_weight_sum"] == debug["Viol_weight_sum"])]
    if not ambiguous.empty:
        print(f"  Removing {len(ambiguous)} ambiguous rows (both Error & Violation, unresolved by count/weight)")
        debug = debug.drop(ambiguous.index)
        y3 = y3.drop(ambiguous.index)
    return y3, debug
 
 
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
 
def coerce_numeric_features(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise KeyError(f"Missing feature column: {c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
 
 
 
def main() -> None:
 
    ensure_dir(CFG.out_dir)

    # Load data
    df = pd.read_csv(CFG.data_file)

    # Expand feature categories to subcategory columns using hfacs_categories (only once)
    feature_cols = []
    for cat in CFG.feature_categories:
        subcats = hfacs_categories.get(cat, [])
        feature_cols.extend(subcats)

    # Coerce numeric features and drop NaNs
    df = coerce_numeric_features(df, tuple(feature_cols))
    before = len(df)
    df = df.dropna(subset=feature_cols)
    after = len(df)
    if after < before:
        print(f" Dropped {before - after} rows due to NaNs in feature columns.")

    # Build 3-class target
    y3, debug = make_three_class_target(
        df=df,
        error_col=CFG.error_target_col,
        viol_col=CFG.viol_target_col,
        tie_break=CFG.tie_break,
    )

    # Count each group after scoring and grouping
    group_counts = y3.value_counts().sort_index()
    print("\nClass distribution after scoring and grouping:")
    for group, count in group_counts.items():
        label = {0: "Neither", 1: "Error", 2: "Violation"}.get(group, str(group))
        print(f"  {label} ({group}): {count}")

    # Save debug assignment file
    debug_path = os.path.join(CFG.out_dir, "svm_debug_assignment.csv")
    debug.to_csv(debug_path, index=False)

    # Features matrix
    X = df.loc[:, feature_cols].astype(float)

    # Balance the groups (simple undersampling for demonstration)
    from sklearn.utils import resample
    min_count = group_counts.min()
    balanced_indices = []
    for group in group_counts.index:
        idx = y3[y3 == group].index
        idx_bal = resample(idx, replace=False, n_samples=min_count, random_state=CFG.seed)
        balanced_indices.extend(idx_bal)
    X_bal = X.loc[balanced_indices].reset_index(drop=True)
    y_bal = y3.loc[balanced_indices].reset_index(drop=True)
    debug_bal = debug.loc[balanced_indices].reset_index(drop=True)

    # Split (stratify by y_bal if possible)
    stratify = y_bal if y_bal.nunique() > 1 else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_bal, y_bal, debug_bal.index,
        test_size=CFG.test_size,
        random_state=CFG.seed,
        stratify=stratify,
    )

    # Model pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=False)),
    ])
    # Grid (poly only; degree only matters for poly)
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": list(CFG.c_grid)},
        {"svc__kernel": ["rbf"], "svc__C": list(CFG.c_grid), "svc__gamma": ["scale", "auto"]},
        {"svc__kernel": ["poly"], "svc__C": list(CFG.c_grid), "svc__degree": list(CFG.degree_grid), "svc__gamma": ["scale", "auto"]},
    ]

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print("\nBest params:", grid.best_params_)
    print("Best CV f1_macro:", grid.best_score_)
    # Test evaluation
    y_pred = best_model.predict(X_test)
    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report (0=Neither, 1=Error, 2=Violation):\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Expose y_test and y_pred as module-level variables for main.py
    import sys
    module = sys.modules[__name__]
    module.y_test = y_test
    module.y_pred = y_pred

    # Save predictions with debug info
    pred_df = pd.DataFrame({
        "index": idx_test,
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    # Fix: Use iloc if idx_test are positions, or ensure debug index matches idx_test
    try:
        debug_test = debug.loc[idx_test].reset_index(drop=True)
    except KeyError:
        # If idx_test are positions, use iloc
        debug_test = debug.iloc[idx_test].reset_index(drop=True)
    pred_df = pd.concat([pred_df.reset_index(drop=True), debug_test], axis=1)
    pred_path = os.path.join(CFG.out_dir, "svm_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print("\nSaved:")
    print(" -", debug_path)
    print(" -", pred_path)
 
if __name__ == "__main__":
    main()
 
 
 
 
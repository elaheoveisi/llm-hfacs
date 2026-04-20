 
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from features.utils import make_three_class_target
from features.balancing import balance_undersample
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
paths_cfg = config_yaml['paths']

class Config:
    data_file: str = paths_cfg['processed_csv']
    out_dir: str = paths_cfg.get('svm_output_dir', './data/processed/svm')
    feature_categories: List[str] = svm_cfg['feature_columns']
    target_columns: List[str] = svm_cfg['target_columns']
    error_target_col: str = svm_cfg.get('error_target_col', 'Error')
    viol_target_col: str = svm_cfg.get('viol_target_col', 'Violation')
    test_size: float = svm_cfg.get('test_size', 0.20)
    seed: int = svm_cfg.get('seed', 7)
    degree_grid: Tuple[int, ...] = tuple(svm_cfg.get('degree_grid', [2, 3]))
    c_grid: Tuple[float, ...] = tuple(svm_cfg.get('c_grid', [0.01, 0.1, 1, 10, 100]))
    tie_break: str = svm_cfg.get('tie_break', 'error')

CFG = Config()

# Weights (defined under svm: in config)
error_weights: Dict[str, float] = svm_cfg.get('error_weights', {})
viol_weights: Dict[str, float] = svm_cfg.get('viol_weights', {})
ERROR_ANOM_COLS: List[str] = list(error_weights.keys())
VIOL_ANOM_COLS: List[str] = list(viol_weights.keys())


 
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
        error_weights=error_weights,
        viol_weights=viol_weights,
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

    combined = X.assign(y3=y3).join(debug.add_prefix("dbg_"))
    combined_bal = balance_undersample(combined, "y3", CFG.seed)
    y_bal = combined_bal.pop("y3")
    debug_bal = combined_bal[[c for c in combined_bal.columns if c.startswith("dbg_")]].rename(columns=lambda c: c[4:])
    X_bal = combined_bal[feature_cols]

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
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)),
    ])
    pipe.fit(X_train, y_train)
    best_model = pipe
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
 
 
 
 
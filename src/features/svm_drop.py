from __future__ import annotations

import os
import sys

_src_dir = os.path.join(os.path.dirname(__file__), "..")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from typing import Dict, List
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features.balancing import balance_undersample
from features.utils import (
    build_labels_from_related_columns_drop_both,
    coerce_numeric_features,
    expand_feature_columns,
)

with open(os.path.join(os.path.dirname(__file__), "../../configs/config.yaml"), "r", encoding="utf-8") as f:
    config_yaml = yaml.safe_load(f)

hfacs_categories = config_yaml["hfacs_categories"]
svm_cfg = config_yaml["svm"]
paths_cfg = config_yaml["paths"]

class Config:
    data_file: str = paths_cfg["processed_csv"]
    out_dir: str = paths_cfg.get("svm_output_dir", "./data/processed/svm").replace("svm", "svm_drop")
    feature_categories: List[str] = svm_cfg["feature_columns"]
    target_columns: List[str] = svm_cfg["target_columns"]
    error_target_col: str = svm_cfg.get("error_target_col", "Error")
    viol_target_col: str = svm_cfg.get("viol_target_col", "Violation")
    test_size: float = svm_cfg.get("test_size", 0.20)
    seed: int = svm_cfg.get("seed", 7)


CFG = Config()

error_weights: Dict[str, float] = hfacs_categories.get("Error", {})
viol_weights: Dict[str, float] = hfacs_categories.get("Violation", {})
ERROR_ANOM_COLS: List[str] = list(error_weights)
VIOL_ANOM_COLS: List[str] = list(viol_weights)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ensure_dir(CFG.out_dir)

    df = pd.read_csv(CFG.data_file)
    feature_cols = expand_feature_columns(CFG.feature_categories, hfacs_categories)

    # Feature columns must exist because they are fed to the model.
    # Related label columns may be missing in some processed datasets; those should
    # behave like all-zero columns rather than aborting the run.
    df = coerce_numeric_features(df, tuple(feature_cols))
    related_present = [c for c in dict.fromkeys(ERROR_ANOM_COLS + VIOL_ANOM_COLS) if c in df.columns]
    for c in related_present:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=feature_cols).copy()
    after = len(df)
    if after < before:
        print(f"\nDropped {before - after} rows due to NaNs in feature columns.")

    y_all, debug_all = build_labels_from_related_columns_drop_both(
        df=df,
        error_related_cols=ERROR_ANOM_COLS,
        violation_related_cols=VIOL_ANOM_COLS,
    )
    keep_mask = y_all >= 0
    dropped_both = int((~keep_mask).sum())

    df = df.loc[keep_mask].copy()
    y = y_all.loc[keep_mask].copy()
    debug = debug_all.loc[keep_mask].copy()
    X = df.loc[:, feature_cols].astype(float)

    print("\nLabeling rule for svm_drop:")
    print("  Error if any Error-related column is 1 and no Violation-related column is 1")
    print("  Violation if any Violation-related column is 1 and no Error-related column is 1")
    print("  Neither if no Error-related or Violation-related columns are 1")
    print("  Both if both groups have any 1, then drop before balancing/modeling")
    print(f"\nDropped {dropped_both} rows labeled as both.")
    print(f"Remaining rows fed to the model: {len(df)}")
    print("\nClass distribution before balancing:")
    for cls, label in [(0, "Neither"), (1, "Error"), (2, "Violation")]:
        print(f"  {label}: {(y == cls).sum()}")

    X["source_index"] = X.index
    combined = X.assign(y3=y)
    combined_bal = balance_undersample(combined, "y3", CFG.seed)
    y_bal = combined_bal.pop("y3")
    source_index = combined_bal.pop("source_index").astype(int)
    X_bal = combined_bal[feature_cols]
    debug_bal = debug.loc[source_index].copy()
    debug_bal.index = X_bal.index

    print(f"\nBalanced size: {len(X_bal)} ({y_bal.value_counts().min()} per class x 3)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal,
        y_bal,
        test_size=CFG.test_size,
        random_state=CFG.seed,
        stratify=y_bal,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report (0=Neither, 1=Error, 2=Violation):\n")
    print(classification_report(y_test, y_pred, digits=4))

    debug_path = os.path.join(CFG.out_dir, "svm_drop_debug_assignment.csv")
    debug.to_csv(debug_path, index_label="index")

    pred_df = debug_bal.loc[X_test.index].copy()
    pred_df.insert(0, "y_pred", y_pred)
    pred_df.insert(0, "y_true", y_test.loc[X_test.index].to_numpy())
    pred_df.insert(0, "index", source_index.loc[X_test.index].to_numpy())
    pred_path = os.path.join(CFG.out_dir, "svm_drop_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path}")


if __name__ == "__main__":
    main()

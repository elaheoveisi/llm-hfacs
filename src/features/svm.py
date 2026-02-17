from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


def _print_table(title: str, rows: List[Dict], columns: List[str]) -> None:
    col_widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(col_widths[c]) for c in columns)
    sep = "-+-".join("-" * col_widths[c] for c in columns)
    print(f"\n{title}")
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(col_widths[c]) for c in columns))


def run_svm_analysis(
    df: pd.DataFrame,
    config_path: str = "./configs/config.yaml",
    n_splits: int = 5,
    seed: int = 7,
    out_csv: str = "data/processed/svm_results.csv",
) -> None:
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["svm"]

    xcols = cfg["feature_columns"]
    ycols = cfg["target_columns"]  # ["Error","Violation"]

    #missing_x = [c for c in xcols if c not in df.columns]
    #missing_y = [c for c in ycols if c not in df.columns]
    #if missing_x or missing_y:
    # raise ValueError(f"Missing columns. X missing: {missing_x} | y missing: {missing_y}")

    X = df[xcols].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    table_cols = ["Target", "Model", "Accuracy", "Precision", "Recall", "F1"]
    rows: List[Dict] = []

    for target in ycols:
        y = df[target].astype(int).to_numpy()
        fold_rows: List[Dict] = []


        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]

            Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)

            clf = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)

            rep = classification_report(yte, pred, output_dict=True, zero_division=0)
            r = {
                "Target": target,
                "Model": f"Fold {fold}",
                "Accuracy": rep["accuracy"],
                "Precision": rep["1"]["precision"],
                "Recall": rep["1"]["recall"],
                "F1": rep["1"]["f1-score"],
            }
            fold_rows.append(r)

        mean_row = {"Target": target, "Model": "Mean"}
        for k in ["Accuracy", "Precision", "Recall", "F1"]:
            mean_row[k] = f"{np.mean([float(fr[k]) for fr in fold_rows]):.4f}"

        rows.extend(fold_rows + [mean_row])

    _print_table("=== SVM (config-driven features) ===", rows, table_cols)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

#print table
def _print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    """Pretty-print a list of row-dicts as a fixed-width table."""
    col_widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(col_widths[c]) for c in columns)
    sep = "-+-".join("-" * col_widths[c] for c in columns)
    print(f"\n{title}")
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(col_widths[c]) for c in columns))

#This dictionary is used to build the results table and CSV output for each SVM fold and the mean.
def _score_row(name: str, ytrue, pred) -> dict:
    """Compute a standard set of metrics for one target."""
    return {
        "Model": name,
        "Accuracy": f"{accuracy_score(ytrue, pred):.4f}",
        "Precision": f"{precision_score(ytrue, pred, zero_division=0):.4f}",
        "Recall": f"{recall_score(ytrue, pred, zero_division=0):.4f}",
        "F1": f"{f1_score(ytrue, pred, zero_division=0):.4f}",
    }


def run_svm_analysis(
    df: pd.DataFrame,
    xcols: list[str],
    ycols: list[str],
    n_splits: int = 5,
    seed: int = 7,
):
    """Run per-label SVM-RBF with SMOTE and stratified k-fold CV.

    Parameters
    ----------
    df : pd.DataFrame
        Row-level dataset with binary indicator columns.
    xcols : list[str]
        Feature column names (Level-2 HFACS categories).
    ycols : list[str]
        Target column names (Level-1 HFACS categories).
    n_splits : int
        Number of stratified k-fold splits.
    seed : int
        Random state for reproducibility.
    """

    # ---------- PREPARE ----------
    df = df.copy()
    df[xcols] = df[xcols].astype(int)
    df[ycols] = df[ycols].astype(int)

    X = df[xcols].to_numpy()
    Y = df[ycols]

    table_cols = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results_dir = "data/processed"
    for target in ycols:
        y = Y[target].to_numpy()

        fold_metrics: list[dict] = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1): #loop iterates over each fold produced by StratifiedKFold
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            # SMOTE on training fold only
            smote = SMOTE(random_state=seed)
            Xtr_res, ytr_res = smote.fit_resample(Xtr, ytr)

            svm = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
            svm.fit(Xtr_res, ytr_res)
            pred = svm.predict(Xte)

            fold_metrics.append(_score_row(f"Fold {fold}", yte, pred))

        # compute mean across folds
        metric_keys = ["Accuracy", "Precision", "Recall", "F1"]
        mean_row = {"Model": "Mean"}
        for k in metric_keys:
            vals = [float(fm[k]) for fm in fold_metrics]
            mean_row[k] = f"{np.mean(vals):.4f}"

        rows = fold_metrics + [mean_row]
        _print_table(f"=== Target: {target} ===", rows, table_cols)

        # Save results to CSV
       

        os.makedirs(results_dir, exist_ok=True)
        out_path = f"{results_dir}/svm_results_{target}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

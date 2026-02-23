from __future__ import annotations
import os
from typing import Dict, List, Tuple 
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

 
 
def _print_table(title: str, rows: List[Dict], columns: List[str]) -> None:
    col_widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(col_widths[c]) for c in columns)
    sep = "-+-".join("-" * col_widths[c] for c in columns)
    print(f"\n{title}")
    print(header)
    print(sep)

    for r in rows:

        print(" | ".join(str(r.get(c, "")).ljust(col_widths[c]) for c in columns))
 
 
def _encode_joint_labels(y_error: np.ndarray, y_violation: np.ndarray) -> np.ndarray:

    """

    Encode (Error, Violation) into one 4-class label:

      (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3

    """

    y_error = y_error.astype(int)

    y_violation = y_violation.astype(int)

    return (y_error * 2 + y_violation).astype(int)  #combine two binary labels into one number
#take two separate yes/no labels and turn them into one single label with 4 possible values.
 
 
def _decode_joint_labels(y_joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    y_joint = y_joint.astype(int)
    y_error = (y_joint // 2).astype(int)
    y_violation = (y_joint % 2).astype(int)
    return y_error, y_violation
 
 
def run_svm_joint_multioutput(
    df: pd.DataFrame,
    config_path: str = "./configs/config.yaml",
    n_splits: int = 5,
    seed: int = 7,
    out_csv: str = "data/processed/svm_joint_results.csv",
) -> None:



    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["svm"]
    xcols = cfg["feature_columns"]
    ycols = cfg["target_columns"]  # expected ["Error", "Violation"] (order matters)
 
    if len(ycols) != 2:
        raise ValueError(f"This joint SVM expects exactly 2 target columns, got: {ycols}")
    err_col, vio_col = ycols[0], ycols[1]

    # Count and print the number in each category for both target columns
    print(f"Counts for {err_col}:")
    print(df[err_col].value_counts())
    print(f"\nCounts for {vio_col}:")
    print(df[vio_col].value_counts())

    # Balance the joint categories (0,1,2,3) to have the same number of samples
    joint_label = df[err_col].astype(int) * 2 + df[vio_col].astype(int)
    min_count = joint_label.value_counts().min()
    balanced_df = (
        df.assign(_joint=joint_label)
        .groupby("_joint", group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=seed))
        .drop(columns=["_joint"])
        .reset_index(drop=True)
    )
    print("[INFO] Joint category counts after balancing:")
    print(balanced_df[[err_col, vio_col]].astype(int).apply(lambda x: x[err_col]*2 + x[vio_col], axis=1).value_counts().sort_index())
    X = balanced_df[xcols].astype(int).to_numpy()
    y_error = balanced_df[err_col].astype(int).to_numpy()
    y_violation = balanced_df[vio_col].astype(int).to_numpy()
    y_joint = _encode_joint_labels(y_error, y_violation)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    table_cols = [
        "Model",
        "JointAcc",
        f"{err_col}_Prec1", f"{err_col}_Rec1", f"{err_col}_F1_1",
        f"{vio_col}_Prec1", f"{vio_col}_Rec1", f"{vio_col}_F1_1",
    ]
    rows: List[Dict] = []
    fold_rows: List[Dict] = []
 
    for fold, (tr, te) in enumerate(skf.split(X, y_joint), 1):

        Xtr, Xte = X[tr], X[te]

        ytr_joint, yte_joint = y_joint[tr], y_joint[te]
 
 
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=seed)

        clf.fit(Xtr, ytr_joint)
 
        pred_joint = clf.predict(Xte)
 
        # Joint accuracy = exact match on the pair (Error,Violation)

        joint_acc = accuracy_score(yte_joint, pred_joint)
 
        # Decode to evaluate each output separately

        yte_err, yte_vio = _decode_joint_labels(yte_joint)

        pr_err, pr_vio = _decode_joint_labels(pred_joint)
 
        rep_err = classification_report(yte_err, pr_err, output_dict=True, zero_division=0)

        rep_vio = classification_report(yte_vio, pr_vio, output_dict=True, zero_division=0)
 
        r = {
            "Model": f"Fold {fold}",
            "JointAcc": f"{joint_acc:.4f}",
            f"{err_col}_Prec1": f"{rep_err['1']['precision']:.4f}",
            f"{err_col}_Rec1": f"{rep_err['1']['recall']:.4f}",
            f"{err_col}_F1_1": f"{rep_err['1']['f1-score']:.4f}",
            f"{vio_col}_Prec1": f"{rep_vio['1']['precision']:.4f}",
            f"{vio_col}_Rec1": f"{rep_vio['1']['recall']:.4f}",
            f"{vio_col}_F1_1": f"{rep_vio['1']['f1-score']:.4f}",
        }
        fold_rows.append(r)

    mean_row: Dict = {"Model": "Mean"}
    for k in table_cols[1:]:
        mean_row[k] = f"{np.mean([float(fr[k]) for fr in fold_rows]):.4f}"
    rows.extend(fold_rows + [mean_row])
 
    _print_table("=== ONE SVM predicting (Error,Violation) jointly (4-class) ===", rows, table_cols)
 
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    

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

    """

    Train ONE SVM model to predict BOTH outputs together by turning (Error, Violation)

    into a single 4-class target, then decoding predictions back to two outputs.
 
    Outputs:

      - Joint accuracy (exact match of both outputs)

      - Per-output precision/recall/F1 for the positive class (1)

      - Also saves fold + mean rows to CSV

    """

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["svm"]
    xcols = cfg["feature_columns"]
    ycols = cfg["target_columns"]  # expected ["Error", "Violation"] (order matters)
 
    if len(ycols) != 2:
        raise ValueError(f"This joint SVM expects exactly 2 target columns, got: {ycols}")
    err_col, vio_col = ycols[0], ycols[1] 
    X = df[xcols].astype(int).to_numpy()
    y_error = df[err_col].astype(int).to_numpy()
    y_violation = df[vio_col].astype(int).to_numpy()
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
 
        # ...existing code...
 
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

            "JointAcc": float(joint_acc),
 
            f"{err_col}_Prec1": float(rep_err["1"]["precision"]),

            f"{err_col}_Rec1": float(rep_err["1"]["recall"]),

            f"{err_col}_F1_1": float(rep_err["1"]["f1-score"]),
 
            f"{vio_col}_Prec1": float(rep_vio["1"]["precision"]),

            f"{vio_col}_Rec1": float(rep_vio["1"]["recall"]),

            f"{vio_col}_F1_1": float(rep_vio["1"]["f1-score"]),

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
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run SVM joint multioutput analysis.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", type=str, default="data/processed/step3_hfacs_categories.csv", help="Path to input CSV data file")
    parser.add_argument("--splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--out", type=str, default="../../data/processed/svm_joint_results.csv", help="Output CSV file")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    run_svm_joint_multioutput(
        df=df,
        config_path=args.config,
        n_splits=args.splits,
        seed=args.seed,
        out_csv=args.out,
    )

if __name__ == "__main__":
    main()

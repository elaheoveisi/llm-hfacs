from __future__ import annotations
import os
import sys

_src_dir = os.path.join(os.path.dirname(__file__), '..')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from features.utils import make_three_class_target_from_config
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

with open(os.path.join(os.path.dirname(__file__), '../../configs/config.yaml'), 'r', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:

    out_dir = _cfg['paths']['decision_tree_output_dir']
    ensure_dir(out_dir)

    dataset_key = _cfg['svm'].get('dataset_key', 'processed_csv')
    df = pd.read_csv(_cfg['paths'][dataset_key])

    _targets = {'Error', 'Violation'}
    feature_cols = [
        col
        for cat, subcats in _cfg['hfacs_categories'].items()
        if cat not in _targets
        for col in subcats
    ]

    y3, debug = make_three_class_target_from_config(df, _cfg)

    group_counts = y3.value_counts().sort_index()
    print("\nClass distribution after scoring and grouping:")
    for group, count in group_counts.items():
        label = {0: "Neither", 1: "Error", 2: "Violation"}.get(group, str(group))
        print(f"  {label} ({group}): {count}")

    debug_path = os.path.join(out_dir, "dt_debug_assignment.csv")
    debug.to_csv(debug_path, index=False)

    X = df.loc[:, feature_cols].astype(float)

    seed = _cfg['svm'].get('seed', 7)
    stratify = y3 if y3.nunique() > 1 else None
    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, y3, debug.index,
        test_size=_cfg['svm'].get('test_size', 0.20),
        random_state=seed,
        stratify=stratify,
    )

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 4, 6, 8, 12, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "ccp_alpha": [0.0, 0.0005, 0.001, 0.005, 0.01],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print("\nBest params:", search.best_params_)
    print("Best CV macro-F1:", round(search.best_score_, 4))

    y_pred = search.best_estimator_.predict(X_test)
    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print("Test macro-F1:", round(f1_score(y_test, y_pred, average="macro"), 4))
    print("\nConfusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report (0=Neither, 1=Error, 2=Violation):\n")
    print(classification_report(y_test, y_pred, digits=4))

    pred_df = pd.DataFrame({
        "index": idx_test,
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    try:
        debug_test = debug.loc[idx_test].reset_index(drop=True)
    except KeyError:
        debug_test = debug.iloc[idx_test].reset_index(drop=True)
    pred_df = pd.concat([pred_df.reset_index(drop=True), debug_test], axis=1)
    pred_path = os.path.join(out_dir, "dt_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print("\nSaved:")
    return y_test, y_pred

if __name__ == "__main__":
    main()

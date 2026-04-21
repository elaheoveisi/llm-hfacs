from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from features.balancing import balance_features_labels
from features.utils import make_three_class_target_from_config

with open(os.path.join(os.path.dirname(__file__), "../../configs/config.yaml"), "r", encoding="utf-8") as f:
    config_yaml = yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    out_dir = config_yaml["paths"].get("random_forest_output_dir", "./data/processed/random_forest")
    ensure_dir(out_dir)

    dataset_key = config_yaml["svm"].get("dataset_key", "processed_csv")
    df = pd.read_csv(config_yaml["paths"][dataset_key])

    _targets = {"Error", "Violation"}
    feature_cols = [
        col
        for cat, subcats in config_yaml["hfacs_categories"].items()
        if cat not in _targets
        for col in subcats
    ]

    y3, _ = make_three_class_target_from_config(df, config_yaml)

    print("\nClass distribution after scoring and grouping:")
    group_counts = y3.value_counts().sort_index()
    for group, count in group_counts.items():
        label = {0: "Neither", 1: "Error", 2: "Violation"}.get(group, str(group))
        print(f"  {label} ({group}): {count}")

    X = df.loc[:, feature_cols].astype(float)
    X_bal, y_bal = balance_features_labels(X, y3)

    stratify = y_bal if y_bal.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal,
        y_bal,
        test_size=0.20,
        random_state=7,
        stratify=stratify,
    )

    model = RandomForestClassifier(random_state=7, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("\nBest params:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report (0=Neither, 1=Error, 2=Violation):\n")
    print(classification_report(y_test, y_pred, digits=4))

    module = sys.modules[__name__]
    module.y_test = y_test
    module.y_pred = y_pred

    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    pred_path = os.path.join(out_dir, "rf_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path}")


if __name__ == "__main__":
    main()

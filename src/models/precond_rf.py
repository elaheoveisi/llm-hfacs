from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from features.balancing import balance_and_report
from features.utils import (
    ensure_dir,
    load_dataset,
    print_eval_metrics,
    save_predictions,
    stratified_split,
)
from models.random_forest import make_four_class_target_from_config


def precond_rf(config):
    """Random Forest using LLM-extracted precondition columns (llm_*) as features."""
    out_dir = config["paths"]["rf_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["rf"]
    random_state = config["models"]["random_state"]

    df = load_dataset(config["paths"]["preconditions_csv"])

    feature_cols = [c for c in df.columns if c.startswith("llm_")]
    if not feature_cols:
        raise ValueError(
            "No llm_* columns found in preconditions_csv. Run extract_preconditions first."
        )

    y4 = make_four_class_target_from_config(df, config)
    label_names = {0: "Neither", 1: "Error", 2: "Violation", 3: "Both"}

    print(f"\n[RF-Preconditions] Using {len(feature_cols)} llm_* feature columns")
    print("\nClass distribution (all data):")
    for cls, name in label_names.items():
        print(f"  {name}: {(y4 == cls).sum()}")

    X = df.loc[:, feature_cols].astype(float)
    X, y4 = balance_and_report(X, y4, "full")
    X_train, X_test, y_train, y_test = stratified_split(
        X, y4, test_size=cfg["test_size"], random_state=random_state
    )

    print("\nClass distribution in train split:")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  Class {cls} ({label_names[cls]}): {count} cases")
    print("\nClass distribution in test split:")
    for cls, count in y_test.value_counts().sort_index().items():
        print(f"  Class {cls} ({label_names[cls]}): {count} cases")

    param_dist = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "min_samples_split": cfg["min_samples_split"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "max_features": cfg["max_features"],
    }

    n_splits = min(cfg["cv_splits"], int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_dist,
        n_iter=cfg["n_iter"],
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV f1_macro: {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    print("\nClassification report (0=Neither, 1=Error, 2=Violation, 3=Both):")
    print_eval_metrics(y_test, y_pred)
    save_predictions(y_test, y_pred, out_dir, "rf_preconditions_predictions.csv")

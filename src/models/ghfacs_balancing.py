from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features.balancing import balance_and_report
from features.utils import (
    build_dataset_XY,
    ensure_dir,
    make_four_class_target,
    print_eval_metrics,
    save_predictions,
    stratified_split,
)

CLASSES = ["AE100 only", "AE200 only", "Both", "Neither"]


def ghfacs_svm(config, df):
    out_dir = config["paths"]["ghfacs_svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["ghfacs_svm"]

    X, y = build_dataset_XY(df, config["ghfacs"]["precondition_cols"], make_four_class_target(df))
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=config["models"]["test_size"], random_state=config["models"]["random_state"]
    )

    assert set(y_train.unique()) == set(y.unique()), (
        f"Missing classes in train split: {set(y.unique()) - set(y_train.unique())}"
    )

    X_train, y_train = balance_and_report(X_train, y_train)
    X_test, y_test = balance_and_report(X_test, y_test)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"], probability=False)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_svm_predictions.csv")


def ghfacs_rf(config, df):
    out_dir = config["paths"]["ghfacs_svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["ghfacs_rf"]
    random_state = config["models"]["random_state"]

    X, y = build_dataset_XY(df, config["ghfacs"]["precondition_cols"], make_four_class_target(df))
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=config["models"]["test_size"], random_state=random_state
    )
    X_train, y_train = balance_and_report(X_train, y_train)

    param_grid = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "min_samples_split": cfg["min_samples_split"],
        "max_features": cfg["max_features"],
    }

    n_splits = min(cfg["cv_splits"], int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV f1_macro: {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_rf_predictions.csv")

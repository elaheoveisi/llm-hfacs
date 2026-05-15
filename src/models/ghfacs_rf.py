from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

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


def ghfacs_rf(config, df):
    out_dir = config["paths"]["ghfacs_svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["ghfacs_rf"]

    X, y = build_dataset_XY(df, config["ghfacs"]["precondition_cols"], make_four_class_target(df))
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=config["models"]["test_size"]
    )
    X_train, y_train = balance_and_report(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        max_features=cfg["max_features"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_rf_predictions.csv")

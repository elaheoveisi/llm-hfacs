from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features.balancing import balance_and_report
from features.utils import (
    ensure_dir,
    filter_tied_rows,
    get_hfacs_feature_cols,
    make_three_class_target_from_config,
    print_eval_metrics,
    save_predictions,
    stratified_split,
)


def svm(config, df):
    out_dir = config["paths"]["svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["svm"]

    feature_cols = [c for c in get_hfacs_feature_cols(config) if c in df.columns]
    y3 = make_three_class_target_from_config(df, config)
    df, y3 = filter_tied_rows(df, y3)

    X = df.loc[:, feature_cols].astype(float)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y3, test_size=cfg["test_size"], random_state=config["models"]["random_state"]
    )
    X_train, y_train = balance_and_report(X_train, y_train)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"], probability=False)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nClassification report (0=Neither, 1=Error, 2=Violation):")
    print_eval_metrics(y_test, y_pred)
    save_predictions(y_test, y_pred, out_dir, "svm_predictions.csv")

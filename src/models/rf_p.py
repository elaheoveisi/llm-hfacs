from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from features.balancing import balance_and_report
from features.utils import (
    ensure_dir,
    filter_tied_rows,
    get_hfacs_feature_cols,
    load_dataset,
    make_three_class_target_from_config,
    print_eval_metrics,
    save_predictions,
    stratified_split,
)

def rf_p(config):
    out_dir = config["paths"]["rf_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["rf"]
    random_state = config["models"]["random_state"]

    df = load_dataset(config["paths"]["processed_csv"])

    feature_cols = [c for c in get_hfacs_feature_cols(config) if c in df.columns]
    y3 = make_three_class_target_from_config(df, config)
    df, y3 = filter_tied_rows(df, y3)

    label_names = {0: "Neither", 1: "Error", 2: "Violation"}
    print("\nClass distribution before balancing (all data):")
    for cls, name in label_names.items():
        print(f"  {name}: {(y3 == cls).sum()}")

    X = df.loc[:, feature_cols].astype(float)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y3, test_size=cfg["test_size"], random_state=random_state
    )
    X_train, y_train = balance_and_report(X_train, y_train)
    X_test, y_test = balance_and_report(X_test, y_test)


    param_grid = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "min_samples_split": cfg["min_samples_split"],
        "max_features": cfg["max_features"],
    }

    n_splits = min(cfg["cv_splits"], int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
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
    print("\nClassification report (0=Neither, 1=Error, 2=Violation):")
    print_eval_metrics(y_test, y_pred)
    save_predictions(y_test, y_pred, out_dir, "rf_p_predictions.csv")

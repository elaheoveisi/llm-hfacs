from __future__ import annotations

import numpy as np
import pymc as pm
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
        X, y, test_size=config["models"]["test_size"]
    )
    X_train, y_train = balance_and_report(X_train, y_train)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"], probability=False)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_svm_predictions.csv")


def ghfacs_svm_nonbalance(config, df):
    out_dir = config["paths"]["ghfacs_svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["ghfacs_svm_nonbalance"]

    X, y = build_dataset_XY(df, config["ghfacs"]["precondition_cols"], make_four_class_target(df))
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=config["models"]["test_size"]
    )
    X_train, y_train = balance_and_report(X_train, y_train)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"], probability=False)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_svm_nonbalance_predictions.csv")



def ghfacs_bayes(config, df):
    out_dir = config["paths"]["ghfacs_svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["ghfacs_bayes"]

    X, y = build_dataset_XY(df, config["ghfacs"]["precondition_cols"], make_four_class_target(df))
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=config["models"]["test_size"]
    )
    X_train, y_train = balance_and_report(X_train, y_train)

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y_train_idx = y_train.map(class_to_idx).values.astype(int)

    X_train_np = X_train.values.astype(float)
    X_test_np = X_test.values.astype(float)
    n_features = X_train_np.shape[1]
    n_classes = len(CLASSES)

    with pm.Model():
        intercepts = pm.Normal("intercepts", mu=0, sigma=1, shape=(n_classes,))
        betas = pm.Normal("betas", mu=0, sigma=1, shape=(n_classes, n_features))

        theta = intercepts + pm.math.dot(X_train_np, betas.T)
        p = pm.Deterministic("p", pm.math.softmax(theta, axis=-1))

        pm.Categorical("y_obs", p=p, observed=y_train_idx)

        trace = pm.sample(
            cfg["n_samples"], tune=cfg["tune"], cores=cfg["cores"],
            progressbar=True,
        )

    betas_samples = trace.posterior["betas"].values.reshape(-1, n_classes, n_features)
    intercepts_samples = trace.posterior["intercepts"].values.reshape(-1, n_classes)

    theta_test = (
        np.einsum("nf,skf->snk", X_test_np, betas_samples)
        + intercepts_samples[:, np.newaxis, :]
    )
    pred_samples = theta_test.argmax(axis=-1)

    y_pred_idx = np.array([
        np.bincount(pred_samples[:, i], minlength=n_classes).argmax()
        for i in range(pred_samples.shape[1])
    ])
    y_pred = [CLASSES[i] for i in y_pred_idx]

    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_bayes_predictions.csv")

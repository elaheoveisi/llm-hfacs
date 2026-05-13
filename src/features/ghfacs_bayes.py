from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.balancing import balance_features_labels
from features.utils import (
    ensure_dir,
    load_config,
    make_four_class_target,
)
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CLASSES = ["AE100 only", "AE200 only", "Both", "anyofthem"]


def ghfacs_bayes():
    config = load_config()

    out_dir = config['paths']['ghfacs_svm_output_dir']
    ensure_dir(out_dir)

    data_dir = config['paths']['ghfacs_data_dir']
    input_file = config['llm']['input']
    df = pd.read_excel(os.path.join(data_dir, input_file))
    limit = config['llm'].get('limit')
    if limit:
        df = df.head(limit)

    precondition_cols = config['ghfacs']['precondition_cols']
    feature_cols = [c for c in precondition_cols if c in df.columns]

    X = df[feature_cols].notna().astype(int)
    y4 = make_four_class_target(df)

    stratify = y4 if y4.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y4,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    print("\nClass distribution before balancing (train only):")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  Class {cls}: {count} cases")

    X_train, y_train = balance_features_labels(X_train, y_train)

    print("\nClass distribution after balancing (train only):")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  Class {cls}: {count} cases")

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

        trace = pm.sample(1000, tune=500, cores=1, random_seed=42, progressbar=True)

# Posterior predictive on test set via all posterior weight samples
    betas_samples = trace.posterior["betas"].values.reshape(-1, n_classes, n_features)  # (S, K, F)
    intercepts_samples = trace.posterior["intercepts"].values.reshape(-1, n_classes)    # (S, K)

    # logits for each posterior sample and each test point: (S, N, K)
    theta_test = np.einsum("nf,skf->snk", X_test_np, betas_samples) + intercepts_samples[:, np.newaxis, :]
    pred_samples = theta_test.argmax(axis=-1)  # (S, N)

    # Point prediction: majority vote across posterior samples
    y_pred_idx = np.array([
        np.bincount(pred_samples[:, i], minlength=n_classes).argmax()
        for i in range(pred_samples.shape[1])
    ])
    y_pred = [CLASSES[i] for i in y_pred_idx]

    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion matrix (rows=true, cols=pred):\n",
          confusion_matrix(y_test, y_pred, labels=CLASSES))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, labels=CLASSES, digits=4))

    module = sys.modules[__name__]
    module.y_test = y_test
    module.y_pred = y_pred

    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    pred_path = os.path.join(out_dir, "ghfacs_bayes_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path}")

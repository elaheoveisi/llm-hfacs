from __future__ import annotations

import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features.balancing import balance_features_labels
from features.utils import (
    ensure_dir,
    filter_tied_rows,
    get_hfacs_feature_cols,
    make_three_class_target_from_config,
)


def svm(config, df):
    out_dir = config["paths"]["svm_output_dir"]
    ensure_dir(out_dir)

    feature_cols = [c for c in get_hfacs_feature_cols(config) if c in df.columns]

    # y3 is the target label for each row
    y3 = make_three_class_target_from_config(df, config)

    df, y3 = filter_tied_rows(df, y3)

    # take all rows (:) but only the feature columns
    X = df.loc[:, feature_cols].astype(float)

    stratify = (
        y3 if y3.nunique() > 1 else None
    )  # check if we have more than 1 class to stratify on
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y3,
        test_size=0.15,
        # random_state=42,
        stratify=stratify,
    )

    print("\nClass distribution before balancing (train only):")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  Class {cls}: {count} cases")

    X_train, y_train = balance_features_labels(X_train, y_train)

    print("\nClass distribution after balancing (train only):")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  Class {cls}: {count} cases")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(C=2.3, kernel="linear", gamma="scale", probability=False)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print(
        "\nConfusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred)
    )
    print("\nClassification report (0=Neither, 1=Error, 2=Violation):\n")
    print(classification_report(y_test, y_pred, digits=4))

    pred_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    )
    pred_path = os.path.join(out_dir, "svm_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path}")

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.balancing import balance_features_labels
from features.utils import (
    ensure_dir,
    load_config,
    load_dataset,
    get_hfacs_feature_cols,
    filter_tied_rows,
    make_three_class_target_from_config,
)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main() -> None:
    config_yaml = load_config()

    out_dir = config_yaml['paths']['svm_output_dir']
    ensure_dir(out_dir)

    df = load_dataset(config_yaml)
    feature_cols = [c for c in get_hfacs_feature_cols(config_yaml) if c in df.columns]

    y3, _ = make_three_class_target_from_config(df, config_yaml)

    df, y3 = filter_tied_rows(df, y3)


    X = df.loc[:, feature_cols].astype(float)

    stratify = y3 if y3.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y3,
        test_size=0.15,
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

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=2.3, kernel="linear", gamma="scale", probability=False)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
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
    pred_path = os.path.join(out_dir, "svm_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path}")

if __name__ == "__main__":
    main()

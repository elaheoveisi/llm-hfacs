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
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def ghfacs_svm():
    config = load_config()

    out_dir = config['paths']['ghfacs_svm_output_dir']
    ensure_dir(out_dir)

    data_dir = config['paths']['ghfacs_data_dir']
    input_file = config['llm']['input']
    df = pd.read_excel(os.path.join(data_dir, input_file))

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

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(gamma="scale", probability=False)),
    ])

    param_grid = {
        "svc__C": [0.01, 0.1, 0.5, 1.0, 2.3, 5.0, 10.0],
        "svc__kernel": ["linear", "rbf"],
    }

    n_splits = min(5, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV f1_macro: {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    classes = ["AE100 only", "AE200 only", "Both", "anyofthem"]
    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion matrix (rows=true, cols=pred):\n",
          confusion_matrix(y_test, y_pred, labels=classes))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, labels=classes, digits=4))

    module = sys.modules[__name__]
    module.y_test = y_test
    module.y_pred = y_pred

    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    pred_path = os.path.join(out_dir, "ghfacs_svm_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path}")

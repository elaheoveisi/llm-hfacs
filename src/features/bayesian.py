"""
One Bayesian Network with:
- category-level DAG nodes
- three separate class nodes in the same DAG:
    Neither, Error_Class, Violation_Class
- scoring/weighting used ONLY to create true 3-class labels
- balancing by undersampling on true 3-class labels
- one DAG
- one 3x3 confusion matrix

Rules:
- no edges among the three class nodes
- no class node -> category node edges
- categories can point to class nodes

Outputs -> ./bn_outputs/
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from pgmpy.estimators import HillClimbSearch, BayesianEstimator
try:
    from pgmpy.estimators import ExpertKnowledge
except Exception:
    ExpertKnowledge = None

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except Exception:
    from pgmpy.models import BayesianNetwork  # type: ignore

from pgmpy.inference import VariableElimination


# -------------------- helpers --------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def uniq(xs: List[str]) -> List[str]:
    return list(dict.fromkeys(xs))


def binarize_series(s: pd.Series, thr: float = 0.0) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce").fillna(0) > thr).astype(int)


def get_bic_score():
    try:
        from pgmpy.estimators import BicScore
        return BicScore
    except Exception:
        from pgmpy.estimators import BIC
        return BIC


# -------------------- scoring/weighting ONLY for labeling --------------------
def _safe_binary_df(df: pd.DataFrame, cols: List[str], thr: float = 0.0) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = binarize_series(df[c], thr) if c in df.columns else 0
    return out


def make_three_class_target(
    df: pd.DataFrame,
    error_col: str,
    viol_col: str,
    error_weights: Dict[str, float],
    viol_weights: Dict[str, float],
    tie_break: str = "error",
    thr: float = 0.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Build true labels only:
      0 = Neither
      1 = Error
      2 = Violation
    """
    if tie_break not in {"error", "violation"}:
        raise ValueError("tie_break must be 'error' or 'violation'")
    if error_col not in df.columns or viol_col not in df.columns:
        raise KeyError(f"Missing target columns: {error_col} / {viol_col}")

    err_flag = binarize_series(df[error_col], thr)
    vio_flag = binarize_series(df[viol_col], thr)

    err_cols = list(error_weights.keys())
    vio_cols = list(viol_weights.keys())

    err_df = _safe_binary_df(df, err_cols, thr)
    vio_df = _safe_binary_df(df, vio_cols, thr)

    err_count = err_df.sum(axis=1)
    vio_count = vio_df.sum(axis=1)

    err_wsum = sum(err_df[c] * w for c, w in error_weights.items()) if err_cols else pd.Series(0, index=df.index)
    vio_wsum = sum(vio_df[c] * w for c, w in viol_weights.items()) if vio_cols else pd.Series(0, index=df.index)

    y3 = pd.Series(0, index=df.index, dtype=int)
    y3[(err_flag == 1) & (vio_flag == 0)] = 1
    y3[(err_flag == 0) & (vio_flag == 1)] = 2

    both = (err_flag == 1) & (vio_flag == 1)
    if both.any():
        # 1. Use anomaly counts
        to_error = both & (err_count > vio_count)
        to_viol  = both & (vio_count > err_count)
        # 2. If counts are tied, use weighted sum
        tie_count = both & (err_count == vio_count)
        to_error |= tie_count & (err_wsum > vio_wsum)
        to_viol  |= tie_count & (vio_wsum > err_wsum)
        # 3. If still tied (counts and weights), use tie-break
        tie_weight = tie_count & (err_wsum == vio_wsum)
        if tie_break == "error":
            to_error |= tie_weight
        else:
            to_viol |= tie_weight
        y3[to_error] = 1
        y3[to_viol] = 2

    debug = pd.DataFrame({
        "Error_flag": err_flag,
        "Violation_flag": vio_flag,
        "Err_anom_count": err_count,
        "Viol_anom_count": vio_count,
        "Err_weight_sum": err_wsum,
        "Viol_weight_sum": vio_wsum,
        "y3": y3,
    })
    # Remove ambiguous cases with zero weighted sum (cannot resolve)
    ambiguous = debug[(debug["Error_flag"] == 1) & (debug["Violation_flag"] == 1) & (debug["Err_weight_sum"] == 0) & (debug["Viol_weight_sum"] == 0)]
    if not ambiguous.empty:
        debug = debug.drop(ambiguous.index)
        y3 = y3.drop(ambiguous.index)
    return y3, debug


# -------------------- category-level DAG inputs --------------------
def build_category_df(
    df: pd.DataFrame,
    hfacs: Dict[str, List[str]],
    category_names: List[str],
    thr: float = 0.0,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for cat in category_names:
        if cat not in hfacs:
            raise KeyError(f"Missing hfacs_categories entry for: {cat}")

        subcols = uniq(hfacs[cat])
        temp = pd.DataFrame(index=df.index)
        for c in subcols:
            temp[c] = binarize_series(df[c], thr) if c in df.columns else 0

        out[cat] = (temp.max(axis=1) > 0).astype(int)
    return out


# -------------------- three separate class nodes --------------------
def add_three_class_nodes(df: pd.DataFrame, y3_col: str = "y3") -> pd.DataFrame:
    out = df.copy()
    out["Neither"] = (out[y3_col] == 0).astype(int)
    out["Error_Class"] = (out[y3_col] == 1).astype(int)
    out["Violation_Class"] = (out[y3_col] == 2).astype(int)
    return out


def decode_three_nodes(df_pred: pd.DataFrame) -> np.ndarray:
    """
    Convert predicted three binary class nodes back to one final class:
      0 = Neither
      1 = Error
      2 = Violation
    """
    arr = df_pred[["Neither", "Error_Class", "Violation_Class"]].to_numpy(dtype=int)

    # If all zero or multiple ones happen, argmax still returns one class.
    # Column order fixes the decoding:
    # 0 -> Neither, 1 -> Error, 2 -> Violation
    return arr.argmax(axis=1)


# -------------------- balancing --------------------
def balance_undersample(df: pd.DataFrame, target_col: str, seed: int) -> pd.DataFrame:
    counts = df[target_col].value_counts().sort_index()
    if len(counts) <= 1:
        return df.copy()

    min_count = int(counts.min())
    parts = [
        df[df[target_col] == cls].sample(n=min_count, replace=False, random_state=seed)
        for cls in counts.index
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# -------------------- BN --------------------
def learn_edges(
    train_bn: pd.DataFrame,
    nodes: List[str],
    class_nodes: List[str],
    category_nodes: List[str],
    max_indegree: Optional[int] = None,
) -> List[Tuple[str, str]]:
    BICScore = get_bic_score()
    scorer = BICScore(train_bn[nodes])
    hc = HillClimbSearch(train_bn[nodes])

    forbidden: List[Tuple[str, str]] = []

    # forbid any class -> category edges
    forbidden += [(c, n) for c in class_nodes for n in category_nodes]

    # forbid any edges among class nodes (both directions)
    for i in range(len(class_nodes)):
        for j in range(len(class_nodes)):
            if i != j:
                forbidden.append((class_nodes[i], class_nodes[j]))

    if ExpertKnowledge is not None:
        expert = ExpertKnowledge(forbidden_edges=forbidden)
        try:
            model = hc.estimate(
                scoring_method=scorer,
                expert_knowledge=expert,
                max_indegree=max_indegree,
                show_progress=False,
            )
        except TypeError:
            try:
                model = hc.estimate(
                    scoring_method=scorer,
                    expert_knowledge=expert,
                    show_progress=False,
                )
            except TypeError:
                model = hc.estimate(
                    scoring_method=scorer,
                    expert_knowledge=expert,
                )
        return list(model.edges())

    # fallback if ExpertKnowledge is unavailable
    try:
        model = hc.estimate(scoring_method=scorer, max_indegree=max_indegree)
    except TypeError:
        model = hc.estimate(scoring_method=scorer)

    forbidden_set = set(forbidden)
    return [e for e in model.edges() if e not in forbidden_set]


def predict_three_nodes(
    bn: BayesianNetwork,
    test_bn: pd.DataFrame,
    evidence_cols: List[str],
    class_nodes: List[str],
) -> pd.DataFrame:
    infer = VariableElimination(bn)
    model_nodes = set(bn.nodes())
    use_evidence = [c for c in evidence_cols if c in model_nodes]

    preds = []
    for _, row in test_bn.iterrows():
        evidence = {c: int(row[c]) for c in use_evidence}
        q = infer.map_query(variables=class_nodes, evidence=evidence, show_progress=False)
        preds.append({c: int(q[c]) for c in class_nodes})

    return pd.DataFrame(preds)


def save_dag(edges: List[Tuple[str, str]], nodes: List[str], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(edges, columns=["parent", "child"]).to_csv(out / "bn_dag_edges.csv", index=False)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.to_pandas_adjacency(G, nodelist=nodes).to_csv(out / "bn_dag_adjacency.csv")

    try:
        from visualization.dag_graph import plot_dag
        plot_dag(G, save_path=str(out / "bn_dag.pdf"))
    except Exception:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(11, 8))
            pos = nx.spring_layout(G, seed=7)
            nx.draw(G, pos, with_labels=True, node_size=1800, font_size=8, arrows=True)
            plt.tight_layout()
            plt.savefig(out / "bn_dag.pdf")
            plt.close()
        except Exception:
            pass


# -------------------- main --------------------
def main() -> None:
    cfg = load_yaml("./configs/config.yaml")
    hfacs = cfg["hfacs_categories"]
    svm_cfg = cfg["svm"]

    out_dir = "./bn_outputs"
    ensure_dir(out_dir)

    data_file = os.path.join("data/processed", svm_cfg.get("data_file", "step3_hfacs_categories.csv"))
    seed = int(svm_cfg.get("seed", 7))
    test_size = float(svm_cfg.get("test_size", 0.20))
    thr = float(svm_cfg.get("bin_threshold", 0.0))

    error_col = svm_cfg.get("error_target_col", "Error")
    viol_col = svm_cfg.get("viol_target_col", "Violation")
    tie_break = svm_cfg.get("tie_break", "error")

    error_weights = cfg.get("error_weights", {})
    viol_weights = cfg.get("viol_weights", {})

    category_nodes = svm_cfg["feature_columns"]
    class_nodes = ["Neither", "Error_Class", "Violation_Class"]

    raw_df = pd.read_csv(data_file)

    # 1) category nodes
    X_cat = build_category_df(raw_df, hfacs, category_nodes, thr=thr)

    # 2) true 3-class labels from scoring/weighting
    if error_col not in raw_df.columns:
        raw_df[error_col] = 0
    if viol_col not in raw_df.columns:
        raw_df[viol_col] = 0

    y3, debug = make_three_class_target(
        df=raw_df,
        error_col=error_col,
        viol_col=viol_col,
        error_weights=error_weights,
        viol_weights=viol_weights,
        tie_break=tie_break,
        thr=thr,
    )

    # 3) create three class nodes
    df = X_cat.copy()
    df["y3"] = y3.astype(int)
    df = add_three_class_nodes(df, y3_col="y3")

    counts_before = df["y3"].value_counts().sort_index()
    print("\nClass distribution before balancing:")
    for k, v in counts_before.items():
        name = {0: "Neither", 1: "Error", 2: "Violation"}[int(k)]
        print(f"  {name} ({int(k)}): {int(v)}")

    # 4) balance on true y3
    df_bal = balance_undersample(df, "y3", seed=seed)

    counts_after = df_bal["y3"].value_counts().sort_index()
    print("\nClass distribution after balancing:")
    for k, v in counts_after.items():
        name = {0: "Neither", 1: "Error", 2: "Violation"}[int(k)]
        print(f"  {name} ({int(k)}): {int(v)}")

    # 5) split
    strat = df_bal["y3"] if df_bal["y3"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df_bal,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    nodes = category_nodes + class_nodes
    train_bn = train_df[nodes].astype(int).copy()
    test_bn = test_df[nodes].astype(int).copy()

    # 6) one DAG + one BN
    edges = learn_edges(
        train_bn=train_bn,
        nodes=nodes,
        class_nodes=class_nodes,
        category_nodes=category_nodes,
        max_indegree=None,
    )

    bn = BayesianNetwork()
    bn.add_nodes_from(nodes)
    bn.add_edges_from(edges)
    bn.fit(train_bn, estimator=BayesianEstimator, prior_type="BDeu")

    save_dag(edges, nodes, out_dir)

    # 7) predict the three class nodes
    pred_nodes_df = predict_three_nodes(
        bn=bn,
        test_bn=test_bn,
        evidence_cols=category_nodes,
        class_nodes=class_nodes,
    )

    # 8) decode back to one class for evaluation
    y_true = test_df["y3"].to_numpy(dtype=int)
    y_pred = decode_three_nodes(pred_nodes_df)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\nBayesian Network results")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion matrix (rows=true, cols=pred) [0,1,2]:\n", cm)
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    # save outputs
    pd.DataFrame(cm, index=["true_0", "true_1", "true_2"], columns=["pred_0", "pred_1", "pred_2"]).to_csv(
        os.path.join(out_dir, "bn_confusion_matrix.csv")
    )
    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "Neither_pred": pred_nodes_df["Neither"],
        "Error_Class_pred": pred_nodes_df["Error_Class"],
        "Violation_Class_pred": pred_nodes_df["Violation_Class"],
    }).to_csv(os.path.join(out_dir, "bn_predictions.csv"), index=False)

    debug.to_csv(os.path.join(out_dir, "bn_debug_assignment.csv"), index=False)

    print("\nSaved:")
    print(" - bn_dag_edges.csv")
    print(" - bn_dag_adjacency.csv")
    print(" - bn_dag.pdf")
    print(" - bn_confusion_matrix.csv")
    print(" - bn_predictions.csv")
    print(" - bn_debug_assignment.csv")


if __name__ == "__main__":
    main()
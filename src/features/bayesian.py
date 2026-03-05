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


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def binarize_series(s: pd.Series, thr: float = 0.0) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce").fillna(0) > thr).astype(int)


def make_three_class_target(
    df: pd.DataFrame, error_col: str, viol_col: str,
    error_weights: Dict[str, float], viol_weights: Dict[str, float],
    thr: float = 0.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build true labels: 0=Neither, 1=Error, 2=Violation.
    Tie-breaking: 1) count, 2) weighted sum, 3) remove if unresolved."""
    if error_col not in df.columns or viol_col not in df.columns:
        raise KeyError(f"Missing target columns: {error_col} / {viol_col}")

    err_flag, vio_flag = binarize_series(df[error_col], thr), binarize_series(df[viol_col], thr)

    def binary_df(weights):
        return pd.DataFrame({c: binarize_series(df[c], thr) if c in df.columns else 0
                             for c in weights}, index=df.index)

    err_df, vio_df = binary_df(error_weights), binary_df(viol_weights)
    err_count, vio_count = err_df.sum(axis=1), vio_df.sum(axis=1)
    err_wsum = sum(err_df[c] * w for c, w in error_weights.items()) if error_weights else pd.Series(0, index=df.index)
    vio_wsum = sum(vio_df[c] * w for c, w in viol_weights.items()) if viol_weights else pd.Series(0, index=df.index)

    y3 = pd.Series(0, index=df.index, dtype=int)
    y3[(err_flag == 1) & (vio_flag == 0)] = 1
    y3[(err_flag == 0) & (vio_flag == 1)] = 2

    both = (err_flag == 1) & (vio_flag == 1)
    if both.any():
        to_error = both & (err_count > vio_count)
        to_viol  = both & (vio_count > err_count)
        tie_count = both & (err_count == vio_count)
        to_error |= tie_count & (err_wsum > vio_wsum)
        to_viol  |= tie_count & (vio_wsum > err_wsum)
        y3[to_error], y3[to_viol] = 1, 2

    debug = pd.DataFrame({
        "Error_flag": err_flag, "Violation_flag": vio_flag,
        "Err_anom_count": err_count, "Viol_anom_count": vio_count,
        "Err_weight_sum": err_wsum, "Viol_weight_sum": vio_wsum, "y3": y3,
    })
    
    # Remove ambiguous cases that couldn't be resolved by count or weighted sum
    ambiguous = debug[(debug["Error_flag"] == 1) & (debug["Violation_flag"] == 1) &
                      (debug["Err_anom_count"] == debug["Viol_anom_count"]) &
                      (debug["Err_weight_sum"] == debug["Viol_weight_sum"])]
    if not ambiguous.empty:
        print(f"  Removing {len(ambiguous)} ambiguous rows (both Error & Violation, unresolved by count/weight)")
        debug, y3 = debug.drop(ambiguous.index), y3.drop(ambiguous.index)
    return y3, debug


def build_category_df(
    df: pd.DataFrame, hfacs: Dict[str, List[str]],
    category_names: List[str], thr: float = 0.0,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for cat in category_names:
        if cat not in hfacs:
            raise KeyError(f"Missing hfacs_categories entry for: {cat}")
        subcols = list(dict.fromkeys(hfacs[cat]))
        temp = pd.DataFrame({c: binarize_series(df[c], thr) if c in df.columns else 0
                             for c in subcols}, index=df.index)
        out[cat] = (temp.max(axis=1) > 0).astype(int)
    return out


def balance_undersample(df: pd.DataFrame, target_col: str, seed: int) -> pd.DataFrame:
    counts = df[target_col].value_counts()
    if len(counts) <= 1:
        return df.copy()
    min_count = int(counts.min())
    parts = [df[df[target_col] == cls].sample(n=min_count, replace=False, random_state=seed)
             for cls in counts.index]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def learn_edges(train_bn: pd.DataFrame, nodes: List[str], class_nodes: List[str],
                category_nodes: List[str], max_indegree: Optional[int] = None) -> List[Tuple[str, str]]:
    try:
        from pgmpy.estimators import BicScore as Scorer
    except Exception:
        from pgmpy.estimators import BIC as Scorer

    scorer = Scorer(train_bn[nodes])
    hc = HillClimbSearch(train_bn[nodes])
    forbidden = ([(c, n) for c in class_nodes for n in category_nodes] +
                 [(class_nodes[i], class_nodes[j]) for i in range(len(class_nodes))
                  for j in range(len(class_nodes)) if i != j])

    if ExpertKnowledge is not None:
        expert = ExpertKnowledge(forbidden_edges=forbidden)
        for kwargs in [{"max_indegree": max_indegree, "show_progress": False},
                       {"show_progress": False}, {}]:
            try:
                return list(hc.estimate(scoring_method=scorer, expert_knowledge=expert, **kwargs).edges())
            except TypeError:
                continue

    for kwargs in [{"max_indegree": max_indegree}, {}]:
        try:
            model = hc.estimate(scoring_method=scorer, **kwargs)
            return [e for e in model.edges() if e not in set(forbidden)]
        except TypeError:
            continue


def predict_three_nodes(
    bn: BayesianNetwork, test_bn: pd.DataFrame,
    evidence_cols: List[str], class_nodes: List[str],
) -> pd.DataFrame:
    infer = VariableElimination(bn)
    use_evidence = [c for c in evidence_cols if c in set(bn.nodes())]
    preds = [
        {c: int(q[c]) for c in class_nodes}
        for _, row in test_bn.iterrows()
        for q in [infer.map_query(variables=class_nodes,
                                  evidence={c: int(row[c]) for c in use_evidence},
                                  show_progress=False)]
    ]
    return pd.DataFrame(preds)


def save_dag(edges: List[Tuple[str, str]], nodes: List[str], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edges, columns=["parent", "child"]).to_csv(out / "bn_dag_edges.csv", index=False)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.to_pandas_adjacency(G, nodelist=nodes).to_csv(out / "bn_dag_adjacency.csv")

    for mod_path in ["visualization.dag_graph", "src.visualization.dag_graph"]:
        try:
            from importlib import import_module
            plot_dag = import_module(mod_path).plot_dag
            break
        except ModuleNotFoundError:
            pass
    else:
        import importlib.util, sys
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../visualization/dag_graph.py"))
        spec = importlib.util.spec_from_file_location("dag_graph", path)
        dag_graph = importlib.util.module_from_spec(spec)
        sys.modules["dag_graph"] = dag_graph
        spec.loader.exec_module(dag_graph)
        plot_dag = dag_graph.plot_dag

    plot_dag(G, save_path=str(out / "bn_dag.pdf"))


def main() -> None:
    cfg = load_yaml("./configs/config.yaml")
    hfacs, svm_cfg, out_dir = cfg["hfacs_categories"], cfg["svm"], "./bn_outputs"
    os.makedirs(out_dir, exist_ok=True)

    data_file = os.path.join("data/processed", svm_cfg.get("data_file", "step3_hfacs_categories.csv"))
    seed       = int(svm_cfg.get("seed", 7))
    test_size  = float(svm_cfg.get("test_size", 0.20))
    thr        = float(svm_cfg.get("bin_threshold", 0.0))
    error_col  = svm_cfg.get("error_target_col", "Error")
    viol_col   = svm_cfg.get("viol_target_col", "Violation")
    error_weights, viol_weights = cfg.get("error_weights", {}), cfg.get("viol_weights", {})
    category_nodes = svm_cfg["feature_columns"]
    class_nodes    = ["Neither", "Error_Class", "Violation_Class"]

    raw_df = pd.read_csv(data_file)
    X_cat  = build_category_df(raw_df, hfacs, category_nodes, thr=thr)

    for col in (error_col, viol_col):
        if col not in raw_df.columns:
            raw_df[col] = 0

    y3, debug = make_three_class_target(raw_df, error_col, viol_col, error_weights, viol_weights, thr)

    df = X_cat.assign(y3=y3.astype(int))
    df["Neither"]        = (df["y3"] == 0).astype(int)
    df["Error_Class"]    = (df["y3"] == 1).astype(int)
    df["Violation_Class"]= (df["y3"] == 2).astype(int)

    names = {0: "Neither", 1: "Error", 2: "Violation"}
    for label, counts in [("before", df), ("after", balance_undersample(df, "y3", seed))]:
        print(f"\nClass distribution {label} balancing:")
        for k, v in counts["y3"].value_counts().sort_index().items():
            print(f"  {names[int(k)]} ({int(k)}): {int(v)}")
        if label == "before":
            df_bal = counts  # reuse after loop
    df_bal = balance_undersample(df, "y3", seed)

    strat = df_bal["y3"] if df_bal["y3"].nunique() > 1 else None
    train_df, test_df = train_test_split(df_bal, test_size=test_size, random_state=seed, stratify=strat)

    nodes    = category_nodes + class_nodes
    train_bn = train_df[nodes].astype(int).copy()
    test_bn  = test_df[nodes].astype(int).copy()

    edges = learn_edges(train_bn, nodes, class_nodes, category_nodes)
    bn = BayesianNetwork()
    bn.add_nodes_from(nodes)
    bn.add_edges_from(edges)
    bn.fit(train_bn, estimator=BayesianEstimator, prior_type="BDeu")
    save_dag(edges, nodes, out_dir)

    pred_nodes_df = predict_three_nodes(bn, test_bn, category_nodes, class_nodes)
    y_true = test_df["y3"].to_numpy(dtype=int)
    y_pred = pred_nodes_df[["Neither", "Error_Class", "Violation_Class"]].to_numpy(dtype=int).argmax(axis=1)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\nBayesian Network results")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion matrix (rows=true, cols=pred) [0,1,2]:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))

    pd.DataFrame(cm, index=["true_0","true_1","true_2"],
                 columns=["pred_0","pred_1","pred_2"]).to_csv(os.path.join(out_dir, "bn_confusion_matrix.csv"))
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred,
                  "Neither_pred": pred_nodes_df["Neither"],
                  "Error_Class_pred": pred_nodes_df["Error_Class"],
                  "Violation_Class_pred": pred_nodes_df["Violation_Class"]
                  }).to_csv(os.path.join(out_dir, "bn_predictions.csv"), index=False)
    debug.to_csv(os.path.join(out_dir, "bn_debug_assignment.csv"), index=False)


if __name__ == "__main__":
    main()
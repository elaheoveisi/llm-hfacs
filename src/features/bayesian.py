from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
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
from project_config import chdir_project_root, get_optional_path, load_config as load_project_config
from features.utils import make_three_class_target_from_config
from features.balancing import balance_undersample


chdir_project_root()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def binarize_series(s: pd.Series, thr: float = 0.0) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce").fillna(0) > thr).astype(int)




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

    raise RuntimeError("Could not learn BN edges with current pgmpy version/settings.")


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


def _load_plot_dag():
    """Import plot_dag from visualization.dag_graph, trying multiple paths."""
    for mod_path in ["visualization.dag_graph", "src.visualization.dag_graph"]:
        try:
            from importlib import import_module
            return import_module(mod_path).plot_dag
        except ModuleNotFoundError:
            continue
    import importlib.util
    import sys
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../visualization/dag_graph.py"))
    spec = importlib.util.spec_from_file_location("dag_graph", path)
    dag_graph = importlib.util.module_from_spec(spec)
    sys.modules["dag_graph"] = dag_graph
    spec.loader.exec_module(dag_graph)
    return dag_graph.plot_dag


def _print_class_dist(df: pd.DataFrame, label: str, class_nodes: Optional[List[str]] = None) -> None:
    names = {i: c for i, c in enumerate(class_nodes)} if class_nodes else {}
    print(f"\nClass distribution {label}:")
    for k, v in df["y3"].value_counts().sort_index().items():
        print(f"  {names.get(int(k), int(k))} ({int(k)}): {int(v)}")


def _save_results(out_dir: str, cm, y_true, y_pred, pred_nodes_df, debug, class_nodes: List[str]) -> None:
    n = len(class_nodes)
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(n)],
                 columns=[f"pred_{i}" for i in range(n)]).to_csv(os.path.join(out_dir, "bn_confusion_matrix.csv"))
    preds = {"y_true": y_true, "y_pred": y_pred}
    preds.update({f"{c}_pred": pred_nodes_df[c] for c in class_nodes})
    pd.DataFrame(preds).to_csv(os.path.join(out_dir, "bn_predictions.csv"), index=False)
    debug.to_csv(os.path.join(out_dir, "bn_debug_assignment.csv"), index=False)


def save_dag(edges: List[Tuple[str, str]], nodes: List[str], out_dir: str, cpds=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edges, columns=["parent", "child"]).to_csv(out / "bn_dag_edges.csv", index=False)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.to_pandas_adjacency(G, nodelist=nodes).to_csv(out / "bn_dag_adjacency.csv")

    plot_dag = _load_plot_dag()
    plot_dag(G, save_path=str(out / "bn_dag.pdf"), cpds=cpds)



# Refactored workflow function
def run_bayesian_workflow(config_path: str = "./configs/config.yaml",
                          data_file: Optional[str] = None, out_dir: Optional[str] = None) -> None:
    cfg = load_project_config(config_path)
    hfacs, svm_cfg = cfg["hfacs_categories"], cfg["svm"]
    default_data_file = str(get_optional_path(cfg, "bn_input_csv", default=cfg["paths"]["processed_csv"]))
    default_out_dir = str(get_optional_path(cfg, "bn_output_dir", default="./data/processed/bayesian"))
    data_file = data_file or default_data_file
    out_dir = out_dir or default_out_dir
    error_col = svm_cfg.get("error_target_col", "Error")
    viol_col = svm_cfg.get("viol_target_col", "Violation")
    error_weights = hfacs.get("Error", {})
    viol_weights = hfacs.get("Violation", {})
    category_nodes = svm_cfg["feature_columns"]
    class_nodes = svm_cfg.get("class_nodes", ["Neither", "Error", "Violation"])
    seed = int(svm_cfg.get("seed", 7))
    test_size = float(svm_cfg.get("test_size", 0.2))
    thr = float(svm_cfg.get("bin_threshold", 0.0))
    prior_type = svm_cfg.get("bn_prior_type", "BDeu")

    os.makedirs(out_dir, exist_ok=True)
    raw_df = pd.read_csv(data_file)
    X_cat  = build_category_df(raw_df, hfacs, category_nodes, thr=thr)

    for col in (error_col, viol_col):
        if col not in raw_df.columns:
            raw_df[col] = 0

    y3, debug = make_three_class_target_from_config(raw_df, cfg)

    dropped = (y3 == -1).sum()
    if dropped:
        print(f"\nDropped full-tie rows: {dropped}")
        mask = y3 != -1
        raw_df = raw_df.loc[mask].copy()
        X_cat = X_cat.loc[mask].copy()
        y3 = y3.loc[mask]

    df = X_cat.assign(y3=y3.astype(int))
    for index, class_name in enumerate(class_nodes):
        df[class_name] = (df["y3"] == index).astype(int)

    _print_class_dist(df, "before balancing", class_nodes)
    df_bal = balance_undersample(df, "y3", seed)
    _print_class_dist(df_bal, "after balancing", class_nodes)

    strat = df_bal["y3"] if df_bal["y3"].nunique() > 1 else None
    train_df, test_df = train_test_split(df_bal, test_size=test_size, random_state=seed, stratify=strat)

    nodes    = category_nodes + class_nodes
    train_bn = train_df[nodes].astype(int).copy()
    test_bn  = test_df[nodes].astype(int).copy()

    edges = learn_edges(train_bn, nodes, class_nodes, category_nodes)
    bn = BayesianNetwork()
    bn.add_nodes_from(nodes)
    bn.add_edges_from(edges)

    bn.fit(train_bn, estimator=BayesianEstimator, prior_type=prior_type)
    save_dag(edges, nodes, out_dir, cpds=bn.get_cpds())

    # Save all CPTs (conditional probability tables) to a CSV file
    cpt_path = os.path.join(out_dir, "bn_cpts.csv")
    with open(cpt_path, "w", encoding="utf-8") as f:
        for cpd in bn.get_cpds():
            f.write(f"CPD of {cpd.variable}\n")
            values = cpd.get_values()
            parents = cpd.get_evidence()
            if parents:
                if hasattr(cpd, "get_cardinality"):
                    cardinality_map = cpd.get_cardinality(parents)
                    parent_cards = [int(cardinality_map[parent]) for parent in parents]
                else:
                    parent_cards = [int(card) for card in cpd.cardinality[1:1 + len(parents)]]
                import itertools
                parent_states = [list(range(int(c))) for c in parent_cards]
                combos = list(itertools.product(*parent_states))
                # Build header row from parent combinations
                header_labels = []
                for combo in combos:
                    label = " | ".join(f"{p}({s})" for p, s in zip(parents, combo))
                    header_labels.append(label)
                f.write(f"{'':30s}" + "".join(f"{h:>20s}" for h in header_labels) + "\n")
                for state_idx in range(len(values)):
                    row_label = f"{cpd.variable}({state_idx})"
                    f.write(f"{row_label:30s}" + "".join(f"{v:>20.6f}" for v in values[state_idx]) + "\n")
            else:
                for state_idx in range(len(values)):
                    f.write(f"{cpd.variable}({state_idx}): {values[state_idx][0]:.6f}\n")
            f.write("\n")
    print(f"Saved CPTs to {cpt_path}")

    pred_nodes_df = predict_three_nodes(bn, test_bn, category_nodes, class_nodes)
    y_true = test_df["y3"].to_numpy(dtype=int)
    y_pred = pred_nodes_df[class_nodes].to_numpy(dtype=int).argmax(axis=1)
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(len(class_nodes))))

    print("\nBayesian Network results")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(f"\nConfusion matrix (rows=true, cols=pred) {list(range(len(class_nodes)))}:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, labels=list(range(len(class_nodes))), target_names=class_nodes, digits=4))

    _save_results(out_dir, cm, y_true, y_pred, pred_nodes_df, debug, class_nodes)

# Example usage:
# run_bayesian_workflow(config_path="./configs/config.yaml")

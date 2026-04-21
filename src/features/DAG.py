from __future__ import annotations
import os
import sys
import yaml
import pandas as pd
from pathlib import Path
import networkx as nx
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

_src_dir = os.path.join(os.path.dirname(__file__), '..')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from causallearn.search.ScoreBased.GES import ges  # noqa: E402
from visualization.dag_graph import plot_dag  # noqa: E402
from features.utils import make_three_class_target_from_config  # noqa: E402
from features.balancing import balance_undersample  # noqa: E402


_cfg_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
with open(_cfg_path, 'r', encoding='utf-8') as _f:
    config_yaml = yaml.safe_load(_f)

_svm = config_yaml.get('svm', {})
_dag = config_yaml.get('dag', {})

CATEGORIES: list[str] = list(config_yaml['hfacs_categories'].keys())
error_weights: dict = config_yaml['hfacs_categories']['Error']
viol_weights: dict = config_yaml['hfacs_categories']['Violation']


def load_hfacs_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df


def _matrix_to_edges(categories, M):
    directed, undirected = [], []
    n = len(categories)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[i, j], M[j, i]
            if a == -1 and b == 1:
                directed.append((categories[i], categories[j]))
            elif a == 1 and b == -1:
                directed.append((categories[j], categories[i]))
            elif a == -1 and b == -1:
                undirected.append((categories[i], categories[j]))
    return directed, undirected


def _orient_undirected_edges_to_dag(G, undirected_edges, sinks):
    for u, v in undirected_edges:
        if G.has_edge(u, v) or G.has_edge(v, u):
            continue
        for a, b in [(u, v), (v, u)]:
            if a in sinks and b not in sinks:
                continue
            G.add_edge(a, b)
            if nx.is_directed_acyclic_graph(G):
                break
            G.remove_edge(a, b)
    return G


def learn_dag_ges(
    df: pd.DataFrame,
    categories: list[str] = CATEGORIES,
    sink_nodes: tuple[str, ...] | None = None,
    max_indegree: int | None = None,
    score_func: str | None = None,
) -> tuple[nx.DiGraph, dict]:
    if sink_nodes is None:
        sink_nodes = tuple(_dag['sink_nodes'])
    if score_func is None:
        score_func = _dag['score_func']
    if max_indegree is None:
        max_indegree = _dag.get('max_indegree')

    X = df[categories].astype(int).to_numpy()
    record = ges(X, score_func=score_func, maxP=max_indegree, parameters=None)
    directed, undirected = _matrix_to_edges(categories, record["G"].graph)

    sinks = set(sink_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(categories)
    G.add_edges_from(directed)

    for s in sinks:
        for other in list(G.successors(s)):
            G.remove_edge(s, other)

    G = _orient_undirected_edges_to_dag(G, undirected, sinks)

    for s in sink_nodes:
        if list(G.successors(s)):
            raise ValueError(f"Sink node {s} has outgoing edges after orientation!")

    return G, record


def export_dag_outputs(G: nx.DiGraph, output_dir: str, data_path: str | None = None) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(list(G.edges()), columns=["parent", "child"]).to_csv(
        outdir / "learned_dag_edges.csv", index=False
    )

    if data_path is not None and Path(data_path).exists():
        df = pd.read_csv(data_path)
        condprobs = [
            {
                "parent": parent,
                "child": child,
                "P(child=1|parent=1)": df[df[parent] == 1][child].mean() if (df[parent] == 1).any() else float("nan"),
            }
            for parent, child in G.edges()
        ]
        pd.DataFrame(condprobs).to_csv(outdir / "learned_dag_conditional_probabilities.csv", index=False)
    else:
        print(f"[WARN] Could not find data file for conditional probabilities: {data_path}")

    nx.to_pandas_adjacency(G, nodelist=list(G.nodes()), weight=None).to_csv(
        outdir / "learned_dag_adjacency_matrix.csv"
    )

    try:
        from networkx.drawing.nx_pydot import write_dot
        write_dot(G, str(outdir / "learned_dag.dot"))
    except Exception:
        pass

    is_dag = nx.is_directed_acyclic_graph(G)
    with open(outdir / "learned_dag_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Nodes: {G.number_of_nodes()}\n")
        f.write(f"Edges: {G.number_of_edges()}\n")
        f.write(f"Is DAG: {is_dag}\n")
        if is_dag:
            f.write("One topological order:\n")
            f.write(" -> ".join(nx.topological_sort(G)) + "\n")


def run_hfacs_causal_learn_ges(
    data_path: str | None = None,
    output_dir: str | None = None,
) -> None:
    data_path = data_path or config_yaml["paths"]["processed_csv"]
    output_dir = output_dir or config_yaml["paths"]["dag_output_dir"]

    df = load_hfacs_data(data_path)
    y3, _ = make_three_class_target_from_config(df, config_yaml)
    df = df[y3 > 0].copy()

    G, record = learn_dag_ges(df)
    export_dag_outputs(G, output_dir, data_path=data_path)
    plot_dag(G, save_path=str(Path(output_dir) / "learned_dag.pdf"))

    score = record.get("score")
    score_str = str(score) if score is not None else "N/A"
    print(f"[INFO] GES complete. Score={score_str}. Outputs in {output_dir}")
    (Path(output_dir) / "learned_dag_score.txt").write_text(f"GES Score: {score_str}\n", encoding="utf-8")


def evaluate_dag_predictions(
    G: nx.DiGraph,
    df: pd.DataFrame,
    tie_break: str = "error",
) -> pd.Series:
    error_parents = list(G.predecessors("Error"))
    viol_parents = list(G.predecessors("Violation"))

    preds = []
    for _, row in df.iterrows():
        pe = any(row.get(p, 0) == 1 for p in error_parents)
        pv = any(row.get(p, 0) == 1 for p in viol_parents)

        if pe and pv:
            preds.append(1 if tie_break == "error" else 2)
        elif pe:
            preds.append(1)
        elif pv:
            preds.append(2)
        else:
            preds.append(0)

    return pd.Series(preds, index=df.index, dtype=int)


def run_dag_evaluation(
    data_path: str | None = None,
    output_dir: str | None = None,
) -> None:
    data_path = data_path or config_yaml["paths"]["processed_csv"]
    output_dir = output_dir or config_yaml["paths"]["dag_output_dir"]

    df = load_hfacs_data(data_path)
    y3, _ = make_three_class_target_from_config(df, config_yaml)

    G, _ = learn_dag_ges(df)

    df_eval = df[CATEGORIES].assign(y3=y3)
    df_bal = balance_undersample(df_eval, "y3", seed=7)
    y3_bal = df_bal.pop("y3")

    y_pred = evaluate_dag_predictions(G, df_bal[CATEGORIES], tie_break=_svm.get("tie_break", "error"))

    labels = [0, 1, 2]
    label_names = ["Neither", "Error", "Violation"]
    cm = confusion_matrix(y3_bal, y_pred, labels=labels)
    acc = accuracy_score(y3_bal, y_pred)

    print("\nDAG prediction results (balanced dataset)")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred) {label_names}:\n{cm}")
    print("\nClassification report:\n")
    print(classification_report(y3_bal, y_pred, labels=labels, target_names=label_names, digits=4))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=[f"true_{n}" for n in label_names],
                 columns=[f"pred_{n}" for n in label_names]).to_csv(out / "dag_confusion_matrix.csv")
    pd.DataFrame({"y_true": y3_bal.values, "y_pred": y_pred.values}).to_csv(
        out / "dag_predictions.csv", index=False
    )
    print(f"\nSaved → {out / 'dag_confusion_matrix.csv'}")


if __name__ == '__main__':
    run_hfacs_causal_learn_ges()
    run_dag_evaluation()

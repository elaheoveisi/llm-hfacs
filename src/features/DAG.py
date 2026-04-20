from __future__ import annotations
import pandas as pd
from pathlib import Path
import networkx as nx

from causallearn.search.ScoreBased.GES import ges
from visualization.dag_graph import plot_dag
from project_config import chdir_project_root, get_optional_path, load_config
from features.utils import make_three_class_target


chdir_project_root()
config_yaml = load_config()

hfacs_categories = config_yaml['hfacs_categories']
dag_cfg = config_yaml.get('dag', {})

# Weights (reuse svm.py logic)
error_weights = config_yaml.get('error_weights', {})
viol_weights = config_yaml.get('viol_weights', {})
ERROR_ANOM_COLS = list(error_weights.keys())
VIOL_ANOM_COLS = list(viol_weights.keys())

def load_hfacs_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df


def get_category_columns(config: dict) -> list[str]:
    # Use all top-level keys in hfacs_categories
    if "hfacs_categories" not in config or not isinstance(config["hfacs_categories"], dict):
        raise KeyError("config must contain a dict key: hfacs_categories")
    return list(config["hfacs_categories"].keys())



def _blacklist_sinks(nodes: list[str], sink_nodes: tuple[str, ...]) -> list[tuple[str, str]]:
    # Prevent any outgoing edge from sink nodes: sink -> other
    return [(s, other) for s in sink_nodes for other in nodes if other != s]




#It takes a table that shows how things are connected and turns it into two lists: one list of arrows (A → B) and one list of simple lines (A — B), depending on the numbers in the table.
def _matrix_to_edges(categories: list[str], M) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    directed: list[tuple[str, str]] = []
    undirected: list[tuple[str, str]] = []
    n = len(categories)

    for i in range(n):
        for j in range(i + 1, n):
            a = M[i, j]
            b = M[j, i]

            if a == -1 and b == 1:
                directed.append((categories[i], categories[j]))  # i -> j
            elif a == 1 and b == -1:
                directed.append((categories[j], categories[i]))  # j -> i
            elif a == -1 and b == -1:
                undirected.append((categories[i], categories[j]))  # i -- j

    return directed, undirected


def _orient_undirected_edges_to_dag(
    G: nx.DiGraph,
    undirected_edges: list[tuple[str, str]],
    sink_nodes: tuple[str, ...],
) -> nx.DiGraph:
    """
    Make a single DAG by orienting undirected edges greedily:
      - never allow sink -> other
      - never create a directed cycle
    """
    sinks = set(sink_nodes)

    for u, v in undirected_edges:
        if G.has_edge(u, v) or G.has_edge(v, u):
            continue

        # Try u->v first, else v->u
        candidates = [(u, v), (v, u)]
        for a, b in candidates:
            # forbid sink outgoing
            if a in sinks and b not in sinks:
                continue

            G.add_edge(a, b)
            if nx.is_directed_acyclic_graph(G):
                break  # keep it
            G.remove_edge(a, b)  # undo and try the other direction

        # If neither direction works (would create cycle), we just skip it.
    return G


def learn_dag_ges(
    df: pd.DataFrame,
    categories: list[str],
    sink_nodes: tuple[str, ...] = ("Error", "Violation"),
    max_indegree: int | None = None,
    score_func: str = "local_score_BDeu",
) -> tuple[nx.DiGraph, dict]:
    """
    Learn an optimized graph using causal-learn GES (score-based search).

    Returns:
      - G: nx.DiGraph (a single DAG we build from the learned CPDAG)
      - record: dict from causal-learn (includes score, graph, etc.)
    """

    X = df[categories].astype(int).to_numpy()

    record = ges(X, score_func=score_func, maxP=max_indegree, parameters=None)

    M = record["G"].graph  # adjacency matrix-like structure
    directed, undirected = _matrix_to_edges(categories, M)

    G = nx.DiGraph()
    G.add_nodes_from(categories)
    G.add_edges_from(directed)

    # Enforce sink constraint: remove any sink outgoing edges
    for s, other in _blacklist_sinks(categories, sink_nodes):
        if G.has_edge(s, other):
            G.remove_edge(s, other)

    # Turn undirected edges into a single DAG (one valid orientation)
    G = _orient_undirected_edges_to_dag(G, undirected, sink_nodes)

    # Final check for sinks
    for s in sink_nodes:
        if list(G.successors(s)):
            raise ValueError(f"Sink node {s} has outgoing edges after orientation!")

    return G, record


# -----------------------------
# Export + Plot
# -----------------------------
def export_dag_outputs(G: nx.DiGraph, output_dir: str, data_path: str | None = None) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Edges
    edges = pd.DataFrame(list(G.edges()), columns=["parent", "child"])
    edges.to_csv(outdir / "learned_dag_edges.csv", index=False)

    source_data_path = Path(data_path) if data_path is not None else None
    if source_data_path is not None and source_data_path.exists():
        df = pd.read_csv(source_data_path)
        condprobs = []
        for parent, child in G.edges():
            parent_1 = df[df[parent] == 1]
            if len(parent_1) > 0:
                p = parent_1[child].mean()
            else:
                p = float("nan")
            condprobs.append({"parent": parent, "child": child, "P(child=1|parent=1)": p})
        pd.DataFrame(condprobs).to_csv(outdir / "learned_dag_conditional_probabilities.csv", index=False)
    else:
        print(f"[WARN] Could not find data file for conditional probabilities: {source_data_path}")

    # Adjacency
    nodes = list(G.nodes())
    A = nx.to_pandas_adjacency(G, nodelist=nodes, weight=None)
    A.to_csv(outdir / "learned_dag_adjacency_matrix.csv")

    # DOT (optional)
    try:
        from networkx.drawing.nx_pydot import write_dot

        write_dot(G, str(outdir / "learned_dag.dot"))
    except Exception:
        pass

    # Summary
    with open(outdir / "learned_dag_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Nodes: {G.number_of_nodes()}\n")
        f.write(f"Edges: {G.number_of_edges()}\n")
        f.write(f"Is DAG: {nx.is_directed_acyclic_graph(G)}\n")
        if nx.is_directed_acyclic_graph(G):
            f.write("One topological order:\n")
            f.write(" -> ".join(list(nx.topological_sort(G))) + "\n")






def run_hfacs_causal_learn_ges(
    config_path: str = "./configs/config.yaml",
    data_path: str | None = None,
    output_dir: str | None = None,
    sink_nodes: tuple[str, ...] = ("Error", "Violation"),
    max_indegree: int | None = None,
    score_func: str = "local_score_BDeu",
) -> None:
    config = load_config(config_path)
    data_path = data_path or str(get_optional_path(config, "dag_input_csv", default=config["paths"]["processed_csv"]))
    output_dir = output_dir or str(get_optional_path(config, "dag_output_dir", default="./data/processed"))
    df = load_hfacs_data(data_path)
    categories = get_category_columns(config)

    # --- Apply three-class scoring/weighting before DAG creation ---
    # You can change these column names if needed
    error_col = config.get('svm', {}).get('error_target_col', 'Error')
    viol_col = config.get('svm', {}).get('viol_target_col', 'Violation')
    tie_break = config.get('svm', {}).get('tie_break', 'error')
    # Add y3 and debug columns to df
    y3, debug = make_three_class_target(df, error_col, viol_col, error_weights, viol_weights, tie_break)
    df['y3'] = y3
    # Use only rows with y3 > 0 (Error or Violation) for DAG learning
    df = df[df['y3'] > 0].copy()

    # Now proceed to DAG learning as before
    G, record = learn_dag_ges(
        df=df,
        categories=categories,
        sink_nodes=sink_nodes,
        max_indegree=max_indegree,
        score_func=score_func,
    )

    export_dag_outputs(G, output_dir, data_path=data_path)
    plot_dag(G, save_path=str(Path(output_dir) / "learned_dag.pdf"))

    score = record.get("score", None)
    score_path = Path(output_dir) / "learned_dag_score.txt"
    if score is not None:
        print(f"[INFO] GES complete. Score={score}. Outputs in {output_dir}")
        with open(score_path, "w", encoding="utf-8") as f:
            f.write(f"GES Score: {score}\n")
    else:
        print(f"[INFO] GES complete. Outputs in {output_dir}")
        with open(score_path, "w", encoding="utf-8") as f:
            f.write("GES Score: N/A\n")

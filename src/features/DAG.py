from __future__ import annotations

import yaml
import pandas as pd
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

# causal-learn (GES)
from causallearn.search.ScoreBased.GES import ges



def load_config(config_path: str = "./configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    # GES is score-based optimization
    # score_func examples: "local_score_BDeu", "local_score_BIC", etc.
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
def export_dag_outputs(G: nx.DiGraph, output_dir: str) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Edges
    edges = pd.DataFrame(list(G.edges()), columns=["parent", "child"])
    edges.to_csv(outdir / "learned_dag_edges.csv", index=False)

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


def plot_dag(
    G: nx.DiGraph,
    save_path: str | None = None,
    layout_seed: int = 42,
    title: str = "Learned HFACS DAG (GES optimized)",
) -> None:

    def get_node_colors(labels):
        palette = [
            "#ffd966", "#a4c2f4", "#b6d7a8", "#f4cccc", "#d9d2e9", "#ffe599",
            "#76a5af", "#e06666", "#6aa84f", "#674ea7", "#c27ba0", "#f6b26b",
            "#8e7cc3", "#cfe2f3", "#ea9999", "#b4a7d6", "#fff2cc", "#b6d7a8",
            "#a2c4c9", "#e69138", "#38761d", "#134f5c", "#990000", "#0b5394"
        ]
        colors = []
        for i, label in enumerate(labels):
            l = label.lower()
            if l == "error":
                colors.append("#003366")
            elif l == "violation":
                colors.append("black")
            else:
                colors.append(palette[i % len(palette)])
        return colors

    pos = nx.circular_layout(G)
    plt.figure(figsize=(12, 8))
    node_labels = list(G.nodes)
    node_colors = get_node_colors(node_labels)

    # Draw all edges with the same thickness
    nx.draw_networkx_edges(
        G,
        pos,
        ax=plt.gca(),
        width=2.0,  # constant thickness for all edges
        edge_color="dimgray",
        arrows=True,
        arrowstyle="-|>",
        min_source_margin=15,
        min_target_margin=15,
    )

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, ax=plt.gca())

    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Circle((0,0), radius=8, color=c, label=l) for c, l in zip(node_colors, node_labels)]
    plt.legend(
        handles=legend_handles,
        labels=node_labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        handletextpad=0.8,
        columnspacing=1.5,
        fontsize=12,
        borderaxespad=0.0
    )
    plt.subplots_adjust(right=0.78)
    plt.title(title)
    plt.subplots_adjust(bottom=0.25)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def run_hfacs_causal_learn_ges(
    config_path: str = "./configs/config.yaml",
    data_path: str = "./data/processed/step3_hfacs_categories.csv",
    output_dir: str = "./data/processed",
    sink_nodes: tuple[str, ...] = ("Error", "Violation"),
    max_indegree: int | None = None,
    score_func: str = "local_score_BDeu",
) -> None:
    config = load_config(config_path)
    df = load_hfacs_data(data_path)
    categories = get_category_columns(config)


    G, record = learn_dag_ges(
        df=df,
        categories=categories,
        sink_nodes=sink_nodes,
        max_indegree=max_indegree,
        score_func=score_func,
    )

    # Assign edge weights based on co-occurrence frequency in the data
    for u, v in G.edges:
        # Count rows where both u and v are 1
        if u in df.columns and v in df.columns:
            freq = int(((df[u] == 1) & (df[v] == 1)).sum())
        else:
            freq = 1
        G[u][v]["weight"] = freq

    export_dag_outputs(G, output_dir)
    plot_dag(G, save_path=str(Path(output_dir) / "learned_dag.pdf"))

    # record often includes a score; keep print safe
    score = record.get("score", None)
    if score is not None:
        print(f"[INFO] GES complete. Score={score}. Outputs in {output_dir}")
    else:
        print(f"[INFO] GES complete. Outputs in {output_dir}")
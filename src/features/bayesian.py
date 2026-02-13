from __future__ import annotations

from typing import Dict, List, Set, Tuple

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def default_hfacs_hierarchy() -> Dict[str, List[str]]:
    return {
        "Organizational_Climate": [
            "Inadequate_Supervision",
            "Failed_to_Correct_Problem",
            "Planned_Inappropriate_Operations",
            "Supervisory_Violation",
        ],
        "Resource_Management/Organizational_Process": [
            "Inadequate_Supervision",
            "Failed_to_Correct_Problem",
            "Planned_Inappropriate_Operations",
            "Supervisory_Violation",
        ],
        "Inadequate_Supervision": ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"],
        "Failed_to_Correct_Problem": ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"],
        "Planned_Inappropriate_Operations": ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"],
        "Supervisory_Violation": ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"],
        "Condition_of_Operators": ["Error", "Violation"],
        "Personnel_Factors": ["Error", "Violation"],
        "Situational_Factors": ["Error", "Violation"],
    }


def allowed_edges_from_hierarchy(h: Dict[str, List[str]]) -> Set[Tuple[str, str]]:
    return {(p, c) for p, children in h.items() for c in children}


def hfacs_layers() -> List[List[str]]:
    return [
        ["Organizational_Climate", "Resource_Management/Organizational_Process"],
        [
            "Inadequate_Supervision",
            "Failed_to_Correct_Problem",
            "Planned_Inappropriate_Operations",
            "Supervisory_Violation",
        ],
        ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"],
        ["Error", "Violation"],
    ]


def layered_positions(layers: List[List[str]], x_gap: float = 3.0, y_gap: float = 2.6):
    pos = {}
    top_y = (len(layers) - 1) * y_gap
    for i, layer_nodes in enumerate(layers):
        y = top_y - i * y_gap
        n = len(layer_nodes)
        x0 = -((n - 1) * x_gap) / 2
        for j, node in enumerate(layer_nodes):
            pos[node] = (x0 + j * x_gap, y)
    return pos


def load_edges_csv(edges_csv: str) -> pd.DataFrame:
    df = pd.read_csv(edges_csv)
    required = {"parent", "child", "N_parent", "N_joint"}
    if not required.issubset(df.columns):
        raise ValueError(f"Need columns: {sorted(required)}")
    df["parent"] = df["parent"].astype(str)
    df["child"] = df["child"].astype(str)
    df["N_parent"] = pd.to_numeric(df["N_parent"], errors="raise").astype(int)
    df["N_joint"] = pd.to_numeric(df["N_joint"], errors="raise").astype(int)
    return df


def bayes_edge_mean(k: int, n: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    if n <= 0:
        return 0.0
    a = alpha + k
    b = beta + (n - k)
    return a / (a + b)


def edge_keep_score(k: int, n: int, alpha: float, beta: float) -> float:
    if n <= 0:
        return -1e9
    w = bayes_edge_mean(k, n, alpha, beta)
    return (w - 0.5) * (n**0.5)


def threshold_prune_edges(
    df_edges: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    min_keep_score: float = -5,
) -> pd.DataFrame:
    work = df_edges.copy()
    work["keep_score"] = work.apply(
        lambda r: edge_keep_score(int(r["N_joint"]), int(r["N_parent"]), alpha, beta),
        axis=1,
    )
    while len(work) > 0:
        idx_min = work["keep_score"].idxmin()
        if float(work.loc[idx_min, "keep_score"]) >= min_keep_score:
            break
        work = work.drop(index=idx_min)
    return work.sort_values(["parent", "child"]).reset_index(drop=True)


def draw_dag_pdf(
    edge_table: pd.DataFrame,
    pdf_path: str,
    title: str,
    weight_col: str = "w_bayes",
    weight_decimals: int = 3,
    min_width: float = 0.8,
    max_width: float = 7.0,
):
    G = nx.DiGraph()
    all_nodes = [node for layer in hfacs_layers() for node in layer]
    G.add_nodes_from(all_nodes)
    for _, r in edge_table.iterrows():
        G.add_edge(r["parent"], r["child"], weight=float(r[weight_col]))

    pos = layered_positions(hfacs_layers())

    missing = [n for n in G.nodes() if n not in pos]
    if missing:
        fb = nx.spring_layout(G, seed=7)
        for n in missing:
            pos[n] = fb[n]

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    if not weights:
        raise ValueError("No edges to draw (edge_table is empty).")

    w_min, w_max = min(weights), max(weights)
    if abs(w_max - w_min) < 1e-12:
        widths = [(min_width + max_width) / 2] * len(weights)
    else:
        widths = [min_width + (w - w_min) / (w_max - w_min) * (max_width - min_width) for w in weights]

    plt.figure(figsize=(16, 10))
    nx.draw_networkx_nodes(G, pos, node_size=2600)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=22,
        width=widths,
        connectionstyle="arc3,rad=0.06",
    )

    edge_labels = {(u, v): f"{d['weight']:.{weight_decimals}f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.55)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

    print("Saved:", pdf_path, "| nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
    print("Is DAG?", nx.is_directed_acyclic_graph(G))



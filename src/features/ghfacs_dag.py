from __future__ import annotations
import os
import sys
import yaml
import pandas as pd
import networkx as nx
from pathlib import Path

_src_dir = os.path.join(os.path.dirname(__file__), '..')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from causallearn.search.ScoreBased.GES import ges
from visualization.dag_graph import plot_dag
from features.DAG import _matrix_to_edges, _orient_undirected_edges_to_dag, export_dag_outputs

_cfg_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
with open(_cfg_path, 'r', encoding='utf-8') as _f:
    _config = yaml.safe_load(_f)

_gdcfg = _config['ghfacs_dag']

"""Loads the Excel file and converts all the GHFACS columns to binary (0/1)"""
def _load_and_binarize(data_path: str, nodes: list[str]) -> pd.DataFrame:
    df = pd.read_excel(data_path)
    missing = [c for c in nodes if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df[nodes].notna().astype(int)


def learn_ghfacs_dag(df: pd.DataFrame) -> tuple[nx.DiGraph, dict]:
    nodes = list(_gdcfg['nodes'])
    sink_nodes = list(_gdcfg['sink_nodes'])
    score_func = _gdcfg['score_func']
    max_indegree = _gdcfg.get('max_indegree')

    X = df[nodes].to_numpy()
    record = ges(X, score_func=score_func, maxP=max_indegree, parameters=None)
    directed, undirected = _matrix_to_edges(nodes, record['G'].graph)

    """Create an empty directed graph"""
    sinks = set(sink_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(directed)

    """Remove any outgoing edges from sink nodes in the undirected set"""
    for s in sinks:
        for other in list(G.successors(s)):
            G.remove_edge(s, other)

    G = _orient_undirected_edges_to_dag(G, undirected, sinks)

    for s in sink_nodes:
        if list(G.successors(s)):
            raise ValueError(f"Sink node {s} has outgoing edges after orientation!")

    return G, record


"""loads the data
binarizes the GHFACS columns
removes empty rows
learns the DAG
exports results
calculates conditional probabilities
plots the DAG
saves the GES score"""



def run_ghfacs_dag(
    data_path: str | None = None,
    output_dir: str | None = None,
) -> None:
    data_path = data_path or os.path.join(
        _config['paths']['ghfacs_data_dir'], _config['llm']['input']
    )
    output_dir = output_dir or _config['paths']['ghfacs_dag_output_dir']

    df = _load_and_binarize(data_path, list(_gdcfg['nodes']))

    # keep only rows with at least one active node
    df = df[df.any(axis=1)].copy()
    print(f"[INFO] Rows with at least one active GHFACS node: {len(df)}")

    G, record = learn_ghfacs_dag(df) 
    """It takes the binary dataframe and returns the learned DAG and the GES record."""

    export_dag_outputs(G, output_dir, data_path=None)

    # conditional probabilities: P(unsafe_act=1 | precondition=1)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    condprobs = []
    for parent, child in G.edges():
        p_child_given_parent = df[df[parent] == 1][child].mean() if (df[parent] == 1).any() else float('nan')
        p_child_baseline = df[child].mean()
        condprobs.append({
            'parent': parent,
            'child': child,
            'P(child=1|parent=1)': round(p_child_given_parent, 4),
            'P(child=1)_baseline': round(p_child_baseline, 4),
            'lift': round(p_child_given_parent / p_child_baseline, 4) if p_child_baseline > 0 else float('nan'),
        })
    pd.DataFrame(condprobs).to_csv(out / 'ghfacs_dag_conditional_probabilities.csv', index=False)

    plot_dag(G, save_path=str(out / 'ghfacs_dag.pdf'))

    score = record.get('score')
    score_str = str(score) if score is not None else 'N/A'
    print(f"[INFO] GES complete. Score={score_str}. Outputs in {output_dir}")
    (out / 'ghfacs_dag_score.txt').write_text(f"GES Score: {score_str}\n", encoding='utf-8')


if __name__ == '__main__':
    run_ghfacs_dag()

"""Excel file
↓
Convert GHFACS labels to 0 and 1
↓
Send this 0/1 table into GES
↓
GES learns possible connections between variables
↓
Code turns those connections into a DAG
↓
Code saves the DAG and probabilities"""
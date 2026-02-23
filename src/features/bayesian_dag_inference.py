import pandas as pd
import numpy as np
from pomegranate.bayesian_network import BayesianNetwork

# -------------------------------------------------
# 1. Load learned DAG edges (CSV must have: parent, child)
# -------------------------------------------------
def load_learned_dag_edges(edges_path):
    df = pd.read_csv(edges_path)
    if df.shape[1] < 2:
        raise ValueError("Edges CSV must have at least two columns: parent, child")
    return list(df.iloc[:, :2].itertuples(index=False, name=None))


# -------------------------------------------------
# 2. Convert edge list -> parent structure (index-based)
# -------------------------------------------------
def edges_to_parent_tuples(nodes, edges):
    index_map = {name: i for i, name in enumerate(nodes)}
    parents = [[] for _ in nodes]

    for parent, child in edges:
        if parent not in index_map or child not in index_map:
            raise ValueError(f"Edge ({parent} -> {child}) not found in node list.")
        parents[index_map[child]].append(index_map[parent])

    return tuple(tuple(p) for p in parents)


# -------------------------------------------------
# 3. Run Bayesian fitting
# -------------------------------------------------
def run_bayesian_inference(data_path, edges_path, nodes):

    # Load data
    data = pd.read_csv(data_path)
    X = data[nodes].to_numpy()

    # Load edges
    edges = load_learned_dag_edges(edges_path)

    # Convert structure
    structure = edges_to_parent_tuples(nodes, edges)

    # Build + Fit model
    model = BayesianNetwork.from_structure(X=X, structure=structure)
    model.fit(X, inertia=0.0)

    return model


# -------------------------------------------------
# 4. Example main execution
# -------------------------------------------------
if __name__ == "__main__":

    # Define your node order EXACTLY as in training
    nodes = [
        "Violation",
        "Situational_Factors",
        "Personnel_Factors",
        "Condition_of_Operators",
        "Inadequate_Supervision",
        "Failed_to_Correct_Problem",
        "Planned_Inappropriate_Operations",
        "Supervisory_Violation",
        "Organizational_Climate",
        "Resource_Management",
        "Error"
    ]

    model = run_bayesian_inference(
        data_path="./data/processed/step3_hfacs_categories.csv",
        edges_path="./outputs/learned_edges.csv",
        nodes=nodes
    )

    print("Model fitted successfully.")
    print("Log-likelihood:", model.log_probability(model.X).sum())

    # Example: Query conditional probability P(Violation | Error=1)
    # Replace 'Error' and 'Violation' with your actual variable names as needed
    evidence = {nodes.index('Error'): 1}  # Set Error=1
    probs = model.predict_proba([evidence])
    print(f"P(Violation | Error=1): {probs[0][nodes.index('Violation')]}")
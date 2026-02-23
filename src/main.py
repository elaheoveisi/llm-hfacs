import pandas as pd
import yaml
import os
from pathlib import Path
import csv
from features.hfacs_order_probability import (
    HFACS_ORDER,
    compute_all_full_hfacs_chains,
    compute_combined_hfacs_matrix,
    compute_hfacs_ordered_probabilities,
)
from utils import skip_run
from features.DAG import run_hfacs_causal_learn_ges





def balance_joint_categories(df, error_col="Error", violation_col="Violation", random_state=42):
    """
    Downsample all (Error, Violation) joint categories to the size of the smallest group.
    Returns a balanced DataFrame.
    """
    joint_label = df[error_col].astype(int) * 2 + df[violation_col].astype(int)
    min_count = joint_label.value_counts().min()
    balanced_df = (
        df.assign(_joint=joint_label)
        .groupby("_joint", group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=random_state))
        .drop(columns=["_joint"])
        .reset_index(drop=True)
    )
    return balanced_df




# Function to run Bayesian DAG processing
def run_bayesian_dag(out_dir):
    """
    Prune Bayesian DAG edges and draw the DAG PDF.
    Args:
        out_dir (Path or str): Output directory for saving results.
    """
    from features import bayesian as bayesian_module
    import pandas as pd
    from pathlib import Path
    out_dir = Path(out_dir)
    df_bayes_path = out_dir / "hfacs_bayesian_dag_edges.csv"
    if not df_bayes_path.exists():
        print(f"[WARN] {df_bayes_path} not found. Skipping Bayesian DAG processing.")
        return
    df_bayes = pd.read_csv(df_bayes_path)
    pruned = bayesian_module.threshold_prune_edges(
        df_bayes,
        alpha=1.0,
        beta=1.0,
        min_keep_score=-3.0,
    )
    pruned.to_csv(out_dir / "hfacs_bayesian_dag_edges_threshold_pruned.csv", index=False)
    bayesian_module.draw_dag_pdf(
        pruned,
        str(out_dir / "bayesian_dag_threshold_pruned.pdf"),
        "Bayesian HFACS DAG (after threshold pruning)",
    )


    # HFACS chain function
    # from models.prediction import compute_full_chain



    # Prune using the function from bayesian.py
    pruned = bayesian_module.threshold_prune_edges(
        df_bayes,
        alpha=1.0,
        beta=1.0,
        min_keep_score=-3.0,
    )

    pruned.to_csv(out_dir / "hfacs_bayesian_dag_edges_threshold_pruned.csv", index=False)

    paths = config["paths"]
    source_columns = config["source_columns"]
    hfacs_map = config["hfacs_categories"]
    processed_dir = os.path.dirname(paths["processed_csv"]) or "./data/processed"
    processed_csv_path = Path(paths["processed_csv"])

    if not processed_csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_csv_path}. "
            "This pipeline is configured to use processed_output.csv as the source dataset."
        )

    print(f"[INFO] Using existing processed dataset: {processed_csv_path}")
    df = read_csv_robust(processed_csv_path)


    with skip_run("run", "load_raw_dataset") as check:
        if check():
            print("[INFO] Skipping raw dataset load (using processed_output.csv).")


    with skip_run("run", "create_hfacs_category_data") as check:
        if check():
            print("[INFO] Skipping HFACS regeneration/writes (using processed_output.csv).")


    with skip_run("run", "hfacs_ordered_probabilities") as check:
        if check():
            print("[INFO] Computing HFACS-ordered conditional probabilities...")
            compute_hfacs_ordered_probabilities(
                df,
                hfacs_order=HFACS_ORDER,
                output_dir="./data/processed",
            )
            print("[INFO] HFACS ordered probability tables saved.")


    with skip_run("run", "hfacs_full_chains") as check:
        if check():
            print("[INFO] Computing ALL HFACS full chains (Error + Violation)...")

            all_chains_df = compute_all_full_hfacs_chains(
                hfacs_order=HFACS_ORDER,
                processed_dir="./data/processed",
            )
            all_chains_df.to_csv(
                "./data/processed/HFACS_all_L4_to_L1_chains.csv",
                index=False,
            )

            print("[INFO] HFACS full-chain computation complete.")
            print(all_chains_df.head())


    with skip_run("run", "hfacs_combined_matrix") as check:
        if check():
            print("[INFO] Computing combined HFACS matrix (L4 → L1)...")

            combined_df = compute_combined_hfacs_matrix(
                hfacs_order=HFACS_ORDER,
                processed_dir="./data/processed",
                filename="HFACS_L4_to_L1_combined.csv",
            )

            print("[INFO] Combined HFACS matrix saved.")
            print(combined_df)



    with skip_run("run", "hfacs_conditional_probabilities") as check:
        if check():
            print("[INFO] Computing conditional probabilities for all HFACS categories...")
            from features.conditional_prob import compute_all_hfacs_probabilities, save_all_conditional_probabilities_to_csv
            results = compute_all_hfacs_probabilities(
                df,
                hfacs_map,
                output_dir=processed_dir,
            )
            save_all_conditional_probabilities_to_csv(results, output_path=f"{processed_dir}/all_conditional_probabilities.csv")
            print(f"[INFO] Conditional probability tables saved to {processed_dir} and all_conditional_probabilities.csv")


    # FINAL: Save learned DAG probability matrix after all other code
    if __name__ == "__main__":
        try:
            from features.DAG import save_dag_probability_matrix
            import networkx as nx
            with skip_run("run", "save_dag_probability_matrix") as check:
                if check():
                    print("[INFO] Saving learned DAG probability matrix...")
                    dag_edges_path = Path(processed_dir) / "learned_dag_edges.csv"
                    dag_nodes = []
                    dag_edges = []
                    if dag_edges_path.exists():
                        df_edges = pd.read_csv(dag_edges_path)
                        dag_edges = list(df_edges.itertuples(index=False, name=None))
                        dag_nodes = list(set([e[0] for e in dag_edges] + [e[1] for e in dag_edges]))
                        G = nx.DiGraph()
                        G.add_nodes_from(dag_nodes)
                        G.add_edges_from(dag_edges)
                        save_dag_probability_matrix(G, df, output_path=str(Path(processed_dir) / "learned_dag_probability_matrix.csv"))
                        print("[INFO] Probability matrix saved as learned_dag_probability_matrix.csv")
                    else:
                        print("[WARN] learned_dag_edges.csv not found, skipping probability matrix save.")

        except Exception as e:
            print(f"[ERROR] Could not save learned DAG probability matrix: {e}")


# HFACS chain function
# from models.prediction import compute_full_chain


def read_csv_robust(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] pandas read_csv failed for {csv_path}: {e}")
        print("[INFO] Retrying with Python csv parser fallback...")
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return pd.DataFrame(rows)


with open("./configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
source_columns = config["source_columns"]
hfacs_map = config["hfacs_categories"]
processed_dir = os.path.dirname(paths["processed_csv"]) or "./data/processed"
processed_csv_path = Path(paths["processed_csv"])

if not processed_csv_path.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {processed_csv_path}. "
        "This pipeline is configured to use processed_output.csv as the source dataset."
    )


print(f"[INFO] Using existing processed dataset: {processed_csv_path}")
df = read_csv_robust(processed_csv_path)

# Count and print the number in each of the four joint categories (Error, Violation)
if "Error" in df.columns and "Violation" in df.columns:
    joint_label = df["Error"].astype(int) * 2 + df["Violation"].astype(int)
    print("[INFO] Counts for each (Error, Violation) joint category:")
    for idx, count in joint_label.value_counts().sort_index().items():
        print(f"  Category {idx} (Error={idx//2}, Violation={idx%2}): {count}")


with skip_run("run", "load_raw_dataset") as check:
    if check():
        print("[INFO] Skipping raw dataset load (using processed_output.csv).")


with skip_run("run", "create_hfacs_category_data") as check:
    if check():
        print("[INFO] Skipping HFACS regeneration/writes (using processed_output.csv).")


with skip_run("run", "hfacs_ordered_probabilities") as check:
    if check():
        print("[INFO] Computing HFACS-ordered conditional probabilities...")
        compute_hfacs_ordered_probabilities(
            df,
            hfacs_order=HFACS_ORDER,
            output_dir="./data/processed",
        )
        print("[INFO] HFACS ordered probability tables saved.")


with skip_run("run", "hfacs_full_chains") as check:
    if check():
        print("[INFO] Computing ALL HFACS full chains (Error + Violation)...")

        all_chains_df = compute_all_full_hfacs_chains(
            hfacs_order=HFACS_ORDER,
            processed_dir="./data/processed",
        )
        all_chains_df.to_csv(
            "./data/processed/HFACS_all_L4_to_L1_chains.csv",
            index=False,
        )

        print("[INFO] HFACS full-chain computation complete.")
        print(all_chains_df.head())


with skip_run("run", "hfacs_combined_matrix") as check:
    if check():
        print("[INFO] Computing combined HFACS matrix (L4 → L1)...")

        combined_df = compute_combined_hfacs_matrix(
            hfacs_order=HFACS_ORDER,
            processed_dir="./data/processed",
            filename="HFACS_L4_to_L1_combined.csv",
        )

        print("[INFO] Combined HFACS matrix saved.")
        print(combined_df)



with skip_run("run", "hfacs_conditional_probabilities") as check:
    if check():
        print("[INFO] Computing conditional probabilities for all HFACS categories...")
        from features.conditional_prob import compute_all_hfacs_probabilities, save_all_conditional_probabilities_to_csv
        results = compute_all_hfacs_probabilities(
            df,
            hfacs_map,
            output_dir=processed_dir,
        )
        save_all_conditional_probabilities_to_csv(results, output_path=f"{processed_dir}/all_conditional_probabilities.csv")
        print(f"[INFO] Conditional probability tables saved to {processed_dir} and all_conditional_probabilities.csv")


with skip_run("skip", "bayesian") as check:
    if check():
        print("[INFO] Running Bayesian HFACS processing...")
        run_bayesian_dag(processed_dir)





with skip_run("skip", "svm") as check:
    if check():
        from features import svm
        # Balance the data before running SVM
        if "Error" in df.columns and "Violation" in df.columns:
            print("[INFO] Balancing joint categories before SVM...")
            df_balanced = balance_joint_categories(df, error_col="Error", violation_col="Violation")
            print("[INFO] New counts after balancing:")
            joint_label_bal = df_balanced["Error"].astype(int) * 2 + df_balanced["Violation"].astype(int)
            for idx, count in joint_label_bal.value_counts().sort_index().items():
                print(f"  Category {idx} (Error={idx//2}, Violation={idx%2}): {count}")
            # Call SVM directly with balanced DataFrame
            svm.run_svm_joint_multioutput(
                df=df_balanced,
                config_path="configs/config.yaml",
                n_splits=5,
                seed=7,
                out_csv="data/processed/svm_joint_results.csv"
            )
        else:
            svm.main()



with skip_run("run", "dag") as check:
    if check():
        print("[INFO] Running DAG discovery with causal-learn...")
        run_hfacs_causal_learn_ges(
            config_path="configs/config.yaml",
            data_path="data/processed/step3_hfacs_categories.csv",
            output_dir="data/processed",
            sink_nodes=("Error", "Violation"),
            score_func="local_score_BDeu",
        )
        print("[INFO] DAG discovery complete.")
